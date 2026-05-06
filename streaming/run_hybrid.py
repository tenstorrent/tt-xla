# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hybrid streaming inference for DeepSeek-V4 / DeepSeek-Pro.

Combines Mode 1 (whole-model compile) with Mode 2 (per-layer execute):

    - Layer-by-layer load + ship onto a *persistent* `model` (weights
      stay device-resident across the entire run).
    - After each layer's ship, run a TRIVIAL EXECUTE (block forward at
      real prefill shape) to trigger PJRT
      `LoadedExecutableInstance::execute → ensure_layout`. That migrates
      the per-shard host data (DistributedHostBuffer / borrowed
      at::Tensor refs) to device, freeing host RAM via RAII (toLayout
      retain=false) and the BufferInstance::fireDoneWithHostBufferEvent
      hook for vanilla torch-xla. Without this dummy execute, host RAM
      accumulates per-layer and OOMs at DeepSeek-Pro scale on a 512 GB
      host (Mode 1's failure mode).
    - Dummy forward writes to mutable KV-state buffers; we re-init
      those (kv_cache, kv_state, score_state) to fresh zeros after the
      flush. Read-only buffers (freqs_cis) are left untouched.
    - After all layers are device-resident, `torch.compile(model,
      backend="tt")` produces ONE whole-model graph for prefill +
      decode (Mode 1 pattern, no per-block dynamo re-trace cost at
      runtime).

Run:
    source venv/activate
    STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 STREAM_NUM_LAYERS=5 \
        python streaming/run_hybrid.py
"""
from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch import nn
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.sparse_mlp import enable_sparse_mlp
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec,
    _block_shard_spec_pro,
    _ship_module_handle_path,
    _strip_cpu_golden_refs,
    _top_level_shard_spec,
    _top_level_shard_spec_pro,
    _upload_with_sharding,
)
from streaming.run_layer_stream import (
    _build_skeleton,
    _ship_top_level,
    _malloc_trim,
    _log,
    _collect_buffer_paths,
    _splice_persistent_buffers,
    _ship_persistent_buffers_raw,
)


# dynamo cache_size_limit: per-layer flush emits one entry per unique
# (block, shape) cache key + whole-model emits 2 (prefill + decode).
# Default 8 is way below 43 layers; bump to 1000.
torch._dynamo.config.cache_size_limit = 1000

# DEBUG_HYBRID_LEAK Compiler fix:
# Disable the TTNNConstEvalInputsToSystemMemory pass so that const-eval
# function inputs (the big stacked expert weights) stay in DEVICE
# storage rather than being moved to system (host) memory by the
# compiler. With expected_layout=DEVICE, ensure_layout actually
# migrates host->device on first use, releasing the multi-device
# DistributedHostBuffer's host data via RAII.
# Gated by env var so we can A/B compare.
_compile_options: Dict[str, object] = {}
if os.environ.get("STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST", "0") == "1":
    _compile_options["enable_const_eval_inputs_to_system_memory"] = False
    print(
        "[hybrid] STREAM_HYBRID_DISABLE_CONSTEVAL_TO_HOST=1 — passing "
        "enable_const_eval_inputs_to_system_memory=false to compiler",
        flush=True,
    )

# IR dump for diagnostics. STREAM_HYBRID_IR_DUMP_DIR=<path> writes
# stage-by-stage .mlir under <path>/irs/ (vhlo, shlo, ttir, ttnn, ...).
# Use with STREAM_NUM_LAYERS=1 to keep dump small. Compares with the
# new-env hang dump to confirm graph identity.
_ir_dump_dir = os.environ.get("STREAM_HYBRID_IR_DUMP_DIR")
if _ir_dump_dir:
    os.makedirs(_ir_dump_dir, exist_ok=True)
    _compile_options["export_path"] = _ir_dump_dir
    print(
        f"[hybrid] STREAM_HYBRID_IR_DUMP_DIR={_ir_dump_dir} — passing "
        f"export_path to compiler (IRs land under {_ir_dump_dir}/irs/)",
        flush=True,
    )

if _compile_options:
    torch_xla.set_custom_compile_options(_compile_options)


# ----------------------------------------------------------------------------
# Config (env vars).
# ----------------------------------------------------------------------------

PROMPT_LEN = int(os.environ.get("STREAM_PROMPT_LEN", "128"))
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "2"))
BATCH_SIZE = int(os.environ.get("STREAM_BATCH_SIZE", "8"))
NUM_LAYERS = int(os.environ.get("STREAM_NUM_LAYERS", "5"))
USE_REALISTIC_INPUTS = os.environ.get("STREAM_USE_REALISTIC", "1") == "1"
PREFILL_FIRST_STEP = os.environ.get("STREAM_PREFILL_FIRST", "1") == "1"
# DEBUG_HYBRID_LEAK: skip the per-layer dummy execute. If RSS still
# climbs ~14 GB/layer without dummy, the leak is in the ship path
# itself, not in the dummy. See streaming/DEBUG_HYBRID_NOTES.md.
SKIP_FLUSH = os.environ.get("STREAM_HYBRID_SKIP_FLUSH", "0") == "1"
# DEBUG_HYBRID_LEAK: force the per-layer dummy graph to lift every
# parameter and buffer in the block as a graph input (by summing them
# into the output). Intent: trigger ensure_layout / per-shard PjrtTensor
# consolidation for each one, so host RAM is released per layer instead
# of accumulating until whole-model execute.
TOUCH_ALL_PARAMS = os.environ.get("STREAM_HYBRID_TOUCH_ALL_PARAMS", "1") == "1"
# Per-layer dummy graph = JUST sum all params/buffers, skip the actual
# block forward. The graph shape doesn't matter for ensure_layout
# migration — only that every param/buffer is a graph input. Saves
# the cost of compiling+executing the full transformer block per
# layer; mutable KV buffers are not touched so no re-init needed.
PARAM_TOUCH_ONLY = (
    os.environ.get("STREAM_HYBRID_PARAM_TOUCH_ONLY", "0") == "1"
)
# DEBUG_HYBRID_LEAK: call torch._dynamo.reset() after each layer flush.
# Intent: drop dynamo cache entries that may hold module references.
DYNAMO_RESET_PER_LAYER = os.environ.get("STREAM_HYBRID_DYNAMO_RESET", "0") == "1"
# DEBUG_HYBRID_LEAK: walk all live Python tensor objects after each
# _log call and report total CPU tensor memory + top-N largest. Tells
# us whether the leak is Python-side (tensors alive somewhere) or
# C++-side (PJRT plugin staging not visible to gc.get_objects).
TRACE_OBJECTS = os.environ.get("STREAM_HYBRID_TRACE_OBJECTS", "0") == "1"
# After every K layers shipped, run a "release execute" — a compiled
# chain forward through layers[0:N] currently shipped — to force PJRT
# `from_pjrt_buffers` consolidation across all those layers' inputs.
# Empirically, single-layer dummy execute releases only ~2 GB/layer
# while a whole-model execute releases ~14 GB/layer. This periodic
# K-layer execute keeps host RAM bounded between ship and final
# whole-model run. 0 = off (per-layer dummy only).
RELEASE_EVERY_K = int(os.environ.get("STREAM_HYBRID_RELEASE_EVERY_K", "0"))
# DEBUG_HYBRID_LEAK Exp A: skip persistent_bufs pre-ship at startup.
# When 1, layer's _buffers are shipped per-layer via
# _ship_module_handle_path along with parameters; mutable buffers are
# re-zeroed after dummy. Tests whether pre-shipping all KV buffers at
# startup contributes to the first-execute fixed cleanup amount.
SKIP_PERSISTENT_BUF_PRESHIP = (
    os.environ.get("STREAM_HYBRID_SKIP_PERSISTENT_BUFS", "0") == "1"
)
# DEBUG_HYBRID_LEAK Exp C-prelude: run a trivial execute (tiny tensor
# add) BEFORE any layer ship, to force mesh-device open and consume
# the "first execute" cleanup amount on a no-op graph. If the
# subsequent first real execute still drops ~67 GB, the trigger is
# NOT mesh-open. If the drop disappears, mesh-open IS the trigger.
PRE_OPEN_MESH = os.environ.get("STREAM_HYBRID_PRE_OPEN_MESH", "0") == "1"
# DEBUG_HYBRID_LEAK Exp C-2: AFTER top-level ship, run a tiny execute
# that touches model.embed.weight (large tensor). If this triggers
# the 67 GB drop, the trigger is "first time a large host tensor
# goes through ensure_layout". Otherwise the trigger is something
# specific to whole-model graphs.
EMBED_ONLY_PRE_EXECUTE = (
    os.environ.get("STREAM_HYBRID_EMBED_ONLY_PRE_EXECUTE", "0") == "1"
)
# Ship-time migration: when on, plugin migrates each shipped tensor
# to device DRAM right after createOwnedHostTensor (in
# BufferInstance::copyFromHostBuffer). Bounds host RAM peak to ~1 layer.
# Requires plugin built with TTPJRT_SHIP_TIME_MIGRATION support and
# the parent mesh device open BEFORE the first ship — otherwise the
# plugin falls back to keeping the host tensor (no harm). To ensure
# mesh is open early, we run a tiny dummy execute before any ship
# when this flag is set.
SHIP_TIME_MIGRATION = (
    os.environ.get("STREAM_HYBRID_SHIP_TIME_MIGRATION", "0") == "1"
)
if SHIP_TIME_MIGRATION:
    # Set the plugin-level env var BEFORE any plugin call so the
    # static-init read inside copyFromHostBuffer picks it up.
    os.environ.setdefault("TTPJRT_SHIP_TIME_MIGRATION", "1")


# ----------------------------------------------------------------------------
# Buffer re-init after dummy forward.
# ----------------------------------------------------------------------------

# Names of buffers that the per-block forward MUTATES. After the dummy
# execute, these have been written with garbage state and must be
# re-zeroed before the real prefill runs. Read-only buffers (freqs_cis,
# etc.) are NOT in this set — leave them alone.
MUTABLE_KV_BUFFER_NAMES = {"kv_cache", "kv_state", "score_state"}


# DEBUG_HYBRID_LEAK: walk gc objects, count CPU tensors and sum bytes.
# Returns (count, total_gb, top_5_descriptions). Used to tell whether
# host RAM is held by live Python tensors vs C++ PJRT plugin staging
# (the latter is invisible to gc.get_objects).
def _tensor_inventory(top_n: int = 5):
    import gc as _gc
    import torch as _t
    cpu_tensors = []
    for obj in _gc.get_objects():
        try:
            if isinstance(obj, _t.Tensor) and obj.device.type == "cpu":
                size = obj.numel() * obj.element_size()
                if size > 0:
                    cpu_tensors.append((size, obj))
        except Exception:
            pass
    cpu_tensors.sort(key=lambda x: x[0], reverse=True)
    total = sum(s for s, _ in cpu_tensors)
    descs = []
    for s, t in cpu_tensors[:top_n]:
        descs.append(f"{tuple(t.shape)} {t.dtype} {s/1e9:.2f}GB")
    return len(cpu_tensors), total / 1e9, descs


def _log_tensor_inventory(tag: str, top_n: int = 5):
    n, gb, descs = _tensor_inventory(top_n=top_n)
    print(
        f"[tensors {tag}] count={n} total={gb:.2f}GB top{top_n}: "
        + " | ".join(descs),
        flush=True,
    )


def _reinit_mutable_kv_buffers(block, persistent_bufs_for_layer, mesh, device):
    """Re-ship fresh zero CPU buffers for each mutable KV-state buffer
    in `block`. Replaces the corrupted device tensors with new zero
    tensors. Updates `block._buffers[name]` AND
    `persistent_bufs_for_layer[full_path]` so the run-time references
    point at the fresh copies.

    Sharding heuristic (dim>=3 → shard axis 0, else replicate) mirrors
    `_ship_persistent_buffers_raw` in streaming_loader.
    """
    for sub, name, full in _collect_buffer_paths(block):
        if name not in MUTABLE_KV_BUFFER_NAMES:
            continue
        b = sub._buffers[name]
        if b is None:
            continue
        if b.dim() >= 3:
            partition_spec = ("_axis_0",) + (None,) * (b.dim() - 1)
        else:
            partition_spec = (None,) * b.dim()
        cpu_zero = torch.zeros(b.shape, dtype=b.dtype)
        xla_t = _upload_with_sharding(
            cpu_zero, mesh, partition_spec, device,
        )
        sub._buffers[name] = xla_t
        persistent_bufs_for_layer[full] = xla_t
        del cpu_zero
    torch_xla.sync(wait=True)


def _block_relative_overrides(weight_overrides: Dict[str, str]) -> Dict[str, str]:
    """Strip `layers.*.` prefix from override patterns so they apply against a
    single `block` instead of the full model. Patterns without the prefix
    (e.g. top-level `embed.weight` or a literal `default`) pass through.
    """
    relative: Dict[str, str] = {}
    layer_prefix = "layers.*."
    for pattern, dtype in weight_overrides.items():
        if pattern.startswith(layer_prefix):
            relative[pattern[len(layer_prefix):]] = dtype
        elif pattern == "default":
            relative["default"] = dtype
    return relative


def make_mesh():
    n = xr.global_runtime_device_count()
    if n == 32:
        mesh_shape = (4, 8)
    elif n == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, n)
    print(f"[mesh] num_devices={n} mesh_shape={mesh_shape}", flush=True)
    return Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1")), mesh_shape


def main(
    weight_overrides: Optional[Dict[str, str]] = None,
    block_shard_spec_fn=None,
    top_level_shard_spec_fn=None,
):
    """Hybrid streaming runner.

    Args:
        weight_overrides: optional dict of `{glob: dtype_str}` for
            `apply_weight_dtype_overrides`. Default `None` keeps weights at
            their native dtype (Flash). Pass the Pro override dict for
            `bfp_bf4`/`bfp_bf8` packing.
        block_shard_spec_fn: factory `(block, mesh) -> Dict[Tensor, Tuple]`
            that returns SPMD shard spec for a single Block. Default
            `_block_shard_spec` (Flash). Pass `_block_shard_spec_pro` for
            Pro's aggressive sharding.
        top_level_shard_spec_fn: factory `(model) -> Dict[Tensor, Tuple]`
            for embed/head/norm/hc_head_* sharding. Default
            `_top_level_shard_spec` (Flash). Pass
            `_top_level_shard_spec_pro` for Pro.
    """
    if block_shard_spec_fn is None:
        block_shard_spec_fn = _block_shard_spec
    if top_level_shard_spec_fn is None:
        top_level_shard_spec_fn = _top_level_shard_spec

    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    device = torch_xla.device()
    bsz = BATCH_SIZE

    # Pre-open mesh device with a tiny execute. Required so that
    # subsequent ships have ClientInstance::parentMesh() set; the
    # plugin's ship-time migration falls back to host-only otherwise.
    # Mesh shape will be the same one make_mesh() chose (4,8 for 32
    # devices, etc.) — driven by the tiny graph's sharding spec.
    if SHIP_TIME_MIGRATION or PRE_OPEN_MESH:
        _log("pre-mesh-open")
        tiny_a = torch.zeros(8, 8, dtype=torch.bfloat16).to(device)
        tiny_b = torch.zeros(8, 8, dtype=torch.bfloat16).to(device)
        xs.mark_sharding(tiny_a, mesh, ("_axis_0", None))
        xs.mark_sharding(tiny_b, mesh, ("_axis_0", None))

        @torch.compile(backend="tt")
        def _tiny_add(a, b):
            return (a + b).sum()

        _ = _tiny_add(tiny_a, tiny_b)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        _log("post-mesh-open")

    # ---- args ----
    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    if NUM_LAYERS < args.n_layers:
        args.n_layers = NUM_LAYERS
        args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    needed = PROMPT_LEN + MAX_NEW_TOKENS
    if max_cr > 0:
        rounded = ((needed + max_cr - 1) // max_cr) * max_cr
        args.max_seq_len = max(rounded, 2 * max_cr)
    else:
        args.max_seq_len = ((needed + 31) // 32) * 32
    print(
        f"[args] n_layers={args.n_layers} bsz={bsz} prompt={PROMPT_LEN} "
        f"max_seq_len={args.max_seq_len} "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    _log("baseline")
    # Pre-open mesh hook moved to the top of main() so that
    # SHIP_TIME_MIGRATION can depend on it. The earlier PRE_OPEN_MESH
    # block here was for an experiment that's now superseded.

    # ---- skeleton + top-level ship ----
    t_section = time.time()
    model = _build_skeleton(args)
    t_skel = time.time() - t_section
    t_ship = time.time()
    _ship_top_level(model, mesh, device,
                    top_level_shard_spec_fn=top_level_shard_spec_fn)
    print(
        f"[step] skeleton+top-level: skeleton={t_skel:.1f}s "
        f"ship={time.time() - t_ship:.1f}s "
        f"total={time.time() - t_section:.1f}s",
        flush=True,
    )
    _log("post-top-level")

    if EMBED_ONLY_PRE_EXECUTE:
        # DEBUG_HYBRID_LEAK Exp C-2: trigger ensure_layout on
        # embed.weight by running a tiny graph that just does an
        # embed lookup. embed is the LARGEST top-level tensor; if
        # the 67 GB drop is tied to the first migration of a large
        # tensor, this should consume it before any layers ship.
        _log("pre-embed-pre-execute")
        ids_small_cpu = torch.zeros(8, 1, dtype=torch.long)
        ids_small = _upload_with_sharding(
            ids_small_cpu, mesh, ("_axis_0", None), device,
        )
        del ids_small_cpu

        @torch.compile(backend="tt")
        def _embed_only(m, ids):
            h = m.embed(ids)
            return h.float().sum()
        _ = _embed_only(model, ids_small)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        _log("post-embed-pre-execute")

    # ---- pre-allocate persistent KV buffers (Mode 2 pattern) ----
    # One-shot ship of zero KV buffers for every layer from a fresh
    # skeleton. These become the persistent KV state for the run.
    persistent_bufs: List[Dict[str, torch.Tensor]] = []
    if not SKIP_PERSISTENT_BUF_PRESHIP:
        print("\n[hybrid] pre-allocating persistent KV buffers ...", flush=True)
        t_section = time.time()
        init_skel = _build_skeleton(args)
        t_skel = time.time() - t_section
        t_ship = time.time()
        for layer_id in range(args.n_layers):
            bufs = _ship_persistent_buffers_raw(
                init_skel.layers[layer_id], mesh, device,
            )
            persistent_bufs.append(bufs)
        del init_skel
        gc.collect()
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        print(
            f"[step] kv-buf-preship: skeleton={t_skel:.1f}s "
            f"ship={time.time() - t_ship:.1f}s "
            f"total={time.time() - t_section:.1f}s",
            flush=True,
        )
        _log("post-init-buffers")
    else:
        # DEBUG_HYBRID_LEAK Exp A: defer KV buffer ship to per-layer path.
        # persistent_bufs[i] populated after each layer's dummy + reinit.
        print(
            "\n[hybrid Exp A] SKIP_PERSISTENT_BUF_PRESHIP=1 — KV buffers "
            "shipped per-layer via _ship_module_handle_path",
            flush=True,
        )
        persistent_bufs = [None] * args.n_layers
        _log("post-init-buffers (skipped)")

    # ---- dummy inputs for per-layer flush ----
    # Use REAL prefill shape (bsz=BATCH_SIZE, seqlen=PROMPT_LEN, sp=0)
    # so the per-layer flush compile produces HLO at the same shape as
    # what whole-model prefill will hit. PJRT in-process compile cache
    # is keyed by HLO; same-cr layers share an HLO so we expect ~3
    # cold compiles + (NUM_LAYERS - 3) cache hits across the flush
    # phase. Same-shape per-block subgraphs may also reduce some XLA
    # internal staging vs. arbitrary dummy shapes.
    dummy_bsz = bsz
    # Dummy flush graph runs block forward to trigger each weight's
    # parametrize-wrapped matmul (so const_eval typecasts bf16 → bfp4
    # and Plan B's retain-clear frees the bf16 input). seqlen MUST
    # match whole-model's PROMPT_LEN — otherwise the const_eval
    # function-body SHA256 ends up subtly different, the cache key
    # misses at whole-model compile time, and the runtime tries to
    # re-run const_eval with the already-freed bf16 input → assertion
    # fail in load_cached. The seqlen=1 attempt was empirically
    # confirmed to break cache reuse despite the body looking
    # identical at the printed-IR level. Override via
    # `STREAM_DUMMY_SEQLEN` only for diagnostic small-shape runs that
    # don't need cache reuse.
    dummy_seqlen = int(os.environ.get("STREAM_DUMMY_SEQLEN",
                                       str(PROMPT_LEN)))
    dim = args.dim
    print(
        f"[hybrid] dummy forward shape: "
        f"bsz={dummy_bsz} seqlen={dummy_seqlen} hc_mult={model.hc_mult} "
        f"dim={dim} (matches real prefill)",
        flush=True,
    )
    h_dummy_cpu = torch.zeros(
        dummy_bsz, dummy_seqlen, model.hc_mult, dim,
        dtype=torch.bfloat16,
    )
    h_dummy = _upload_with_sharding(
        h_dummy_cpu, mesh, ("_axis_0", None, None, None), device,
    )
    sp_dummy = torch.tensor(0, dtype=torch.long).to(device)
    ids_dummy_cpu = torch.zeros(dummy_bsz, dummy_seqlen, dtype=torch.long)
    ids_dummy = _upload_with_sharding(
        ids_dummy_cpu, mesh, ("_axis_0", None), device,
    )
    del h_dummy_cpu, ids_dummy_cpu
    torch_xla.sync(wait=True)

    if PARAM_TOUCH_ONLY:
        # Skip the actual block forward — just sum every param/buffer
        # so each one becomes a graph input that fires ensure_layout
        # (= host->device migration) at execute time. The compile is
        # tiny (no attention / MoE / collectives) and the executable
        # never writes to mutable KV buffers, so re-init isn't needed.
        @torch.compile(backend="tt")
        def run_block_flush(block, h, sp, ids):
            s = h.float().sum()
            for p in block.parameters():
                s = s + p.float().sum()
            for b in block.buffers():
                s = s + b.float().sum()
            return s
    elif TOUCH_ALL_PARAMS:
        # DEBUG_HYBRID_LEAK: lift every parameter and buffer of the
        # block as a graph input by summing them into the output. Each
        # `p.float().sum()` is a real XLA op that reads the full tensor,
        # so dynamo is forced to add p as a graph leaf and PJRT's
        # `from_pjrt_buffers` consolidation fires for it during execute.
        # Without this, dummy was leaving most params in per-shard
        # plugin staging until whole-model execute.
        @torch.compile(backend="tt")
        def run_block_flush(block, h, sp, ids):
            out = block(h, sp, ids)
            s = out.float().sum()
            for p in block.parameters():
                s = s + p.float().sum()
            for b in block.buffers():
                s = s + b.float().sum()
            return s
    else:
        @torch.compile(backend="tt")
        def run_block_flush(block, h, sp, ids):
            return block(h, sp, ids)

    # K-layer release execute. Compiles a *partial whole-model* through
    # `embed → layers[0:end_id] → norm → head` for each (end_id,
    # seqlen) pair. Empirically the host-staging release fires only on
    # the FIRST release execute (when embed.weight & friends migrate to
    # device for the first time). Subsequent releases hit early-return
    # in ensure_layout and don't trigger cleanup. To force fresh
    # ensure_layout migrations on every release we:
    #   (1) vary `seqlen` (cycled within [PROMPT_LEN, max_seq_len]) so
    #       dynamo recompiles with different input shape;
    #   (2) build NEW input BufferInstances per call (fresh CPU
    #       upload), giving fresh PJRT tensor handles.
    _release_cache: Dict[tuple, "callable"] = {}
    _release_call_idx = [0]
    # Seqlen pool: PROMPT_LEN, PROMPT_LEN+stride, ..., bounded by
    # max_seq_len and >= max_cr. Cycles for many releases.
    _seqlen_pool = []
    _stride = max(32, max_cr) if max_cr > 0 else 32
    _candidate = PROMPT_LEN
    while _candidate <= args.max_seq_len:
        if _candidate >= (max_cr if max_cr else 1):
            _seqlen_pool.append(_candidate)
        _candidate += _stride
    if not _seqlen_pool:
        _seqlen_pool = [PROMPT_LEN]
    print(
        f"[hybrid] release seqlen pool: {_seqlen_pool} "
        f"(cycled per release)",
        flush=True,
    )

    def _run_release_execute(end_id: int) -> None:
        idx = _release_call_idx[0]
        _release_call_idx[0] += 1
        seqlen = _seqlen_pool[idx % len(_seqlen_pool)]
        cache_key = (end_id, seqlen)
        if cache_key in _release_cache:
            fn = _release_cache[cache_key]
        else:
            def partial_fwd(input_ids, sp):
                h = model.embed(input_ids)
                h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
                for i in range(end_id):
                    h = model.layers[i](h, sp, input_ids)
                h = model.norm(h)
                logits = model.head(
                    h,
                    model.hc_head_fn,
                    model.hc_head_scale,
                    model.hc_head_base,
                    model.norm,
                )
                return logits.float().sum()
            fn = torch.compile(partial_fwd, backend="tt")
            _release_cache[cache_key] = fn

        # Create fresh input tensors per call — new BufferInstances
        # mean ensure_layout sees them as first-use even if dynamo
        # cache-hits on shape.
        ids_cpu = torch.zeros(bsz, seqlen, dtype=torch.long)
        fresh_ids = _upload_with_sharding(
            ids_cpu, mesh, ("_axis_0", None), device,
        )
        fresh_sp = torch.tensor(0, dtype=torch.long).to(device)
        del ids_cpu

        print(
            f"[hybrid] partial release execute "
            f"(embed→layers[0:{end_id}]→norm→head, seqlen={seqlen}) ...",
            flush=True,
        )
        t0 = time.time()
        _ = fn(fresh_ids, fresh_sp)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        print(
            f"[hybrid] partial release execute layers[0:{end_id}] "
            f"seqlen={seqlen} done in {time.time() - t0:.1f}s",
            flush=True,
        )

    # ---- per-layer ship + dummy-execute flush loop ----
    for layer_id in range(args.n_layers):
        t_layer = time.time()
        cr = args.compress_ratios[layer_id]
        print(
            f"\n[hybrid] === layer {layer_id}/{args.n_layers - 1} cr={cr} ===",
            flush=True,
        )
        block = model.layers[layer_id]

        # 1. Load HF weights for this block.
        t_load = time.time()
        block_sd = weight_loader.load_block_state_dict(layer_id)
        prefix = f"layers.{layer_id}."
        stripped = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in block_sd.items()
        }
        block.load_state_dict(stripped, strict=False)
        del block_sd, stripped
        gc.collect()
        _log(f"l{layer_id} post-load")
        if TRACE_OBJECTS:
            _log_tensor_inventory(f"l{layer_id} post-load")

        # 2. Sparse-MLP rewrite + strip CPU goldens.
        enable_sparse_mlp(
            block, mesh=mesh_shape, cluster_axis=0, config=args, verbose=False,
        )
        if TRACE_OBJECTS:
            _log_tensor_inventory(f"l{layer_id} post-sparse-pre-strip")
        _strip_cpu_golden_refs(block)
        gc.collect()
        _log(f"l{layer_id} post-sparse")
        if TRACE_OBJECTS:
            _log_tensor_inventory(f"l{layer_id} post-sparse")
        t_load = time.time() - t_load

        # 3. Splice persistent KV (device tensors) into block._buffers.
        # block now has device-resident KV buffers + CPU params.
        # Exp A path: persistent_bufs[layer_id] is None — leave block's
        # CPU buffers in place; _ship_module_handle_path will ship them
        # along with parameters.
        if persistent_bufs[layer_id] is not None:
            _splice_persistent_buffers(block, persistent_bufs[layer_id])

        # 4. Ship CPU params to device via handle path. The
        # _ship_module_handle_path skips already-device buffers from
        # step 3, so we only ship parameters here.
        t_ship = time.time()
        block_specs = block_shard_spec_fn(block, mesh)
        block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
        del block_specs
        _ship_module_handle_path(
            block, block_specs_by_id, mesh, device,
            verbose=False, tag=f"block-{layer_id}",
        )
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        t_ship = time.time() - t_ship
        _log(f"l{layer_id} post-ship")
        if TRACE_OBJECTS:
            _log_tensor_inventory(f"l{layer_id} post-ship")

        # 4b. Apply per-layer weight dtype overrides BEFORE the dummy flush
        # so each per-layer flush graph traces with weight_dtype_override ops
        # → const_eval typecasts bf16 → bfp_bf4/bfp_bf8 at flush-compile time
        # → packed weights cached on device per layer (instead of only at
        # whole-model compile, by which time 60 layers of bf16 expert weights
        # would already have OOM'd device DRAM).
        # Done AFTER ship/mark_sharding (line 587) so parametrization wraps
        # the device-resident sharded original.
        # Done AFTER `_block_shard_spec` (which uses tensor identity) — that
        # ran inside `_ship_module_handle_path` already.
        if weight_overrides:
            applied_block = apply_weight_dtype_overrides(
                block, _block_relative_overrides(weight_overrides)
            )
            if layer_id == 0:
                # Log only once — pattern is the same for every layer.
                print(
                    f"[wdtype] per-layer apply: {len(applied_block)} overrides "
                    f"per block",
                    flush=True,
                )
                for path, dtype_str in applied_block:
                    print(f"[wdtype]   {path} -> {dtype_str}", flush=True)

        # 5. DUMMY EXECUTE: triggers PJRT
        # LoadedExecutableInstance::execute → ensure_layout → release
        # of plugin-owned host staging buffers. Same shape as real
        # prefill so PJRT compile cache keys align.
        # Force replicated output so the all_gather is fused into the
        # forward graph (Mode 2 pattern; required to avoid a separate
        # reader-side compile that would defeat the in-process cache).
        t_flush = time.time()
        if SKIP_FLUSH:
            # Debug mode: skip the dummy flush entirely. Used to compare
            # ship-only vs ship+flush RSS patterns.
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
        else:
            hook = block.register_forward_hook(
                sharding_constraint_hook(block, mesh, (None, None, None, None))
            )
            try:
                _ = run_block_flush(block, h_dummy, sp_dummy, ids_dummy)
                torch_xla.sync(wait=True)
                xm.wait_device_ops()
            finally:
                hook.remove()
            # 6. Re-init mutable KV buffers (kv_cache, kv_state,
            # score_state) that the dummy forward corrupted. freqs_cis
            # etc. are read-only and untouched.
            # Skip when PARAM_TOUCH_ONLY: that mode never runs the
            # block forward, so KV buffers stay clean.
            if not PARAM_TOUCH_ONLY:
                # Exp A path: persistent_bufs[layer_id] is None on entry;
                # _reinit_mutable_kv_buffers writes to a fresh dict that we
                # then assign back to populate persistent_bufs[i].
                if persistent_bufs[layer_id] is None:
                    persistent_bufs[layer_id] = {}
                _reinit_mutable_kv_buffers(
                    block, persistent_bufs[layer_id], mesh, device,
                )
                torch_xla.sync(wait=True)
                xm.wait_device_ops()
        t_flush = time.time() - t_flush
        gc.collect()
        _malloc_trim()

        t_total = time.time() - t_layer
        print(
            f"[hybrid l{layer_id} cr={cr}] "
            f"total={t_total:.1f}s load={t_load:.1f}s "
            f"ship={t_ship:.1f}s flush={t_flush:.1f}s",
            flush=True,
        )
        _log(f"l{layer_id} post-flush")
        if TRACE_OBJECTS:
            _log_tensor_inventory(f"l{layer_id} post-flush")
        if DYNAMO_RESET_PER_LAYER:
            # DEBUG_HYBRID_LEAK: drop all dynamo compilation cache
            # entries. If RSS drops noticeably afterward, dynamo's cache
            # was holding tensor references.
            torch._dynamo.reset()
            gc.collect()
            _malloc_trim()
            _log(f"l{layer_id} post-dynamo-reset")
            if TRACE_OBJECTS:
                _log_tensor_inventory(f"l{layer_id} post-dynamo-reset")

        # K-layer release execute: every K layers, run a chain forward
        # through layers[0:layer_id+1] to force PJRT consolidation
        # (releases per-shard plugin host staging that single-layer
        # dummy doesn't fully release).
        if RELEASE_EVERY_K > 0 and (layer_id + 1) % RELEASE_EVERY_K == 0:
            end_id = layer_id + 1
            _run_release_execute(end_id)
            # Chain forward writes to all touched layers' kv_cache,
            # kv_state, score_state. Re-zero them so real prefill starts
            # from clean state.
            for j in range(end_id):
                _reinit_mutable_kv_buffers(
                    model.layers[j], persistent_bufs[j], mesh, device,
                )
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
            gc.collect()
            _malloc_trim()
            _log(f"post-release-at-{end_id}")
            if TRACE_OBJECTS:
                _log_tensor_inventory(f"post-release-at-{end_id}")

    # ---- all layers device-resident; whole-model compile + run ----
    print("\n[hybrid] all layers device-resident. Whole-model compile ...",
          flush=True)
    _log("post-all-layers")
    if TRACE_OBJECTS:
        _log_tensor_inventory("post-all-layers")

    # Weight dtype overrides are applied PER-LAYER in the streaming loop
    # above (after each block's ship, before its dummy flush) — see step 4b
    # for the rationale. Doing it here (post all-layers) was the original
    # design but only triggered bf16 → bfp_bf4 typecast at whole-model
    # compile, by which time 60 layers of bf16 expert weights had already
    # OOM'd device DRAM. Per-layer apply ensures each flush's const_eval
    # caches the packed weight before the next ship adds more pressure.

    head_hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(head_hook)

    # DEBUG_HYBRID_LEAK: split logs around compile to localize the
    # 70 GB RSS drop (compile vs execute). See DEBUG_HYBRID_NOTES.md.
    _log("pre-torch-compile")
    t_section = time.time()
    compiled = torch.compile(model, backend="tt")
    print(
        f"[step] torch.compile (lazy wrap, no XLA compile yet): "
        f"{time.time() - t_section:.2f}s",
        flush=True,
    )
    _log("post-torch-compile")

    # ---- inputs ----
    if USE_REALISTIC_INPUTS:
        from tests.torch.models.deepseek_v4 import realistic_inputs
        prompt_ids, _ = realistic_inputs.get_realistic_inputs(
            layer_id=args.n_hash_layers, batch_size=bsz, seq_len=PROMPT_LEN,
        )
        prompt_ids = prompt_ids.contiguous()
        print(
            f"[input] using realistic_inputs (layer={args.n_hash_layers}) "
            f"shape={tuple(prompt_ids.shape)}",
            flush=True,
        )
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
        pad_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        canned = [
            "How are you today?",
            "What is the capital of France?",
            "Explain machine learning briefly.",
            "Who painted the Mona Lisa?",
        ]
        rows = []
        for i in range(bsz):
            prompt = canned[i % len(canned)]
            ids = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False,
            ).input_ids[0]
            if ids.shape[0] >= PROMPT_LEN:
                ids = ids[-PROMPT_LEN:]
            else:
                pad = torch.full(
                    (PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long
                )
                ids = torch.cat([pad, ids], dim=0)
            rows.append(ids)
        prompt_ids = torch.stack(rows, dim=0).contiguous()

    # ---- prefill + decode ----
    generated: List[List[int]] = [[] for _ in range(bsz)]
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(0, dtype=torch.long).to(device)

    # Optional background decode pre-compile.
    # When STREAM_HYBRID_BG_DECODE_COMPILE=1, spawn a thread that calls
    # `compiled(fake_decode_input)` in parallel with main thread's prefill.
    # This overlaps the COLD COMPILE phases of prefill and decode graphs;
    # the actual XLA executes still run, so KV state IS mutated by the bg
    # call. After bg finishes, mutable KV buffers are re-zeroed. If bg's
    # exec finishes before main's exec starts (typical when bg compile <
    # main compile), no race; otherwise output is non-deterministic
    # (correctness intentionally not required for this experiment).
    bg_thread = None
    bg_started_at = None
    bg_done = None
    if os.environ.get("STREAM_HYBRID_BG_DECODE_COMPILE", "0") == "1":
        import threading as _threading
        bg_done = _threading.Event()
        # Build fake decode input on device once on main thread, then
        # hand off to bg thread (avoids cross-thread tensor allocation).
        fake_token_cpu = torch.zeros(bsz, 1, dtype=torch.long)
        fake_token = _upload_with_sharding(
            fake_token_cpu, mesh, ("_axis_0", None), device,
        )
        del fake_token_cpu
        fake_sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
        torch_xla.sync(wait=True)

        def _bg_compile_decode():
            try:
                # Separate torch.compile wrapper for the bg thread:
                # dynamo's _ModuleStackTracer / fake-tensor proxy state
                # is per-OptimizedModule, so two threads sharing the
                # same wrapper would race ('not tracked with proxy'
                # crash). model params are still the same nn.Parameter
                # instances shared with `compiled`, and the resulting
                # HLO is identical, so PJRT in-process compile cache
                # hits when main calls `compiled` later.
                bg_compiled = torch.compile(model, backend="tt")
                t = time.time()
                _ = bg_compiled(fake_token, fake_sp)
                torch_xla.sync(wait=True)
                xm.wait_device_ops()
                t_compile_exec = time.time() - t
                # Re-zero mutable KV buffers corrupted by bg's decode exec.
                t = time.time()
                for layer_id, block in enumerate(model.layers):
                    if persistent_bufs[layer_id] is not None:
                        _reinit_mutable_kv_buffers(
                            block, persistent_bufs[layer_id], mesh, device,
                        )
                torch_xla.sync(wait=True)
                t_reinit = time.time() - t
                print(
                    f"[bg] decode pre-compile: compile+exec="
                    f"{t_compile_exec:.1f}s reinit_kv={t_reinit:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"[bg] FAILED: {e}", flush=True)
            finally:
                bg_done.set()

        bg_started_at = time.time()
        bg_thread = _threading.Thread(target=_bg_compile_decode, daemon=True)
        bg_thread.start()
        print(
            "[bg] decode pre-compile thread launched in parallel with prefill",
            flush=True,
        )

    _log("pre-prefill")
    t0 = time.time()
    prefill_logits = compiled(prompt_ids_tt, sp_tt)
    # DEBUG_HYBRID_LEAK: extra log to split lazy-IR construction (return
    # of compiled call) vs actual XLA compile+execute (sync).
    _log("post-prefill-call-pre-sync")
    torch_xla.sync(wait=True)
    print(
        f"[prefill] compile+exec {time.time() - t0:.1f}s",
        flush=True,
    )
    _log("post-prefill")
    # DEBUG_HYBRID_LEAK: run prefill a second time with fresh inputs
    # to test whether subsequent executes can also release host data
    # or only the first one does.
    if os.environ.get("STREAM_HYBRID_DEBUG_DOUBLE_PREFILL", "0") == "1":
        prompt_ids_tt2 = prompt_ids.clone().to(device)
        xs.mark_sharding(prompt_ids_tt2, mesh, ("_axis_0", None))
        sp_tt2 = torch.tensor(0, dtype=torch.long).to(device)
        _log("pre-prefill-2")
        t0 = time.time()
        _ = compiled(prompt_ids_tt2, sp_tt2)
        torch_xla.sync(wait=True)
        print(
            f"[prefill-2] compile+exec {time.time() - t0:.1f}s",
            flush=True,
        )
        _log("post-prefill-2")
    if TRACE_OBJECTS:
        _log_tensor_inventory("post-prefill")

    # Join bg pre-compile thread (if any) before decode loop. By this
    # point bg should be long done since its compile time < prefill
    # compile time and it runs in parallel.
    if bg_thread is not None:
        t_join = time.time()
        bg_done.wait()
        bg_thread.join()
        print(
            f"[bg] joined after prefill, wait={time.time() - t_join:.2f}s, "
            f"bg total wall={time.time() - bg_started_at:.1f}s",
            flush=True,
        )

    next_ids = prefill_logits.detach().to("cpu").argmax(dim=-1)
    for i in range(bsz):
        generated[i].append(int(next_ids[i].item()))
    print(f"[prefill] first ids[:8]={next_ids[:8].tolist()}", flush=True)

    prev_token = next_ids.unsqueeze(1)
    for step in range(MAX_NEW_TOKENS - 1):
        sp_value = PROMPT_LEN + step
        prev_token_tt = prev_token.to(device)
        xs.mark_sharding(prev_token_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(sp_value, dtype=torch.long).to(device)
        t0 = time.time()
        decode_logits = compiled(prev_token_tt, sp_tt)
        # Lazy IR construction returns fast; force the actual XLA
        # compile (first step only) + execute by syncing.
        torch_xla.sync(wait=True)
        elapsed = time.time() - t0
        next_ids = decode_logits.detach().to("cpu").argmax(dim=-1)
        for i in range(bsz):
            generated[i].append(int(next_ids[i].item()))
        # Step 0 = cold compile + exec; subsequent steps = exec only
        # (PJRT in-process compile cache hits since shape is fixed).
        kind = "compile+exec" if step == 0 else "exec"
        print(
            f"[decode {step + 1}] sp={sp_value} {kind}={elapsed:.2f}s "
            f"ids[:8]={next_ids[:8].tolist()}",
            flush=True,
        )
        prev_token = next_ids.unsqueeze(1)

    print("\n[done] generated tokens (first 4 rows):", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
        for i in range(min(4, bsz)):
            ids = generated[i]
            joined = tokenizer.decode(ids)
            print(f"  [{i}] ids={ids}", flush=True)
            print(f"      joined={joined!r}", flush=True)
    except Exception as e:
        print(f"  (tokenizer decode failed: {e}; raw ids only)", flush=True)
        for i in range(min(4, bsz)):
            print(f"  [{i}] {generated[i]}", flush=True)


if __name__ == "__main__":
    main()
