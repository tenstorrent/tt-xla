# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming loader for DeepSeek-V4-Flash.

Loads HF weights one block at a time, ships each block to the TT mesh,
frees host-side storage, and moves on to the next block. Replaces the
all-at-once `_build_and_load_full_model` path used by
`tests/torch/models/deepseek_v4/test_deepseek_v4_full_e2e.py`.

See ../DESIGN.md for the rationale.
"""

from __future__ import annotations

import gc
import os
from typing import Dict, Tuple

import psutil
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
from torch import nn

from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from tests.torch.models.deepseek_v4 import weight_loader
from tt_torch.sparse_mlp import enable_sparse_mlp


# ----------------------------------------------------------------------------
# Memory tracing helper. Call at every phase boundary so peak RAM
# regressions are visible without re-instrumenting every time.
# ----------------------------------------------------------------------------


def _rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def _meminfo() -> str:
    """Detailed memory snapshot. RSS alone can be misleading because the
    glibc allocator may pool freed pages without returning to the OS;
    this also surfaces VmData (heap+anon size) and the system's free
    memory so we can tell whether the memory is actually live or just
    pooled."""
    p = psutil.Process(os.getpid())
    info = p.memory_info()
    sysm = psutil.virtual_memory()
    rss = info.rss / 1e9
    vms = info.vms / 1e9
    sys_avail = sysm.available / 1e9
    sys_used = sysm.used / 1e9
    return f"rss={rss:.2f} vms={vms:.2f} sys_used={sys_used:.2f} sys_avail={sys_avail:.2f} GB"


# Lazy libc handle for malloc_trim. RSS measurements lie if Python's
# allocator pools hold freed pages without returning them to the OS;
# malloc_trim(0) forces a release so RSS reflects actual live storage.
_LIBC = None


def _malloc_trim() -> None:
    global _LIBC
    try:
        import ctypes
        if _LIBC is None:
            _LIBC = ctypes.CDLL("libc.so.6")
        _LIBC.malloc_trim(0)
    except Exception:
        pass  # best-effort; not all platforms have it


def _log_mem(tag: str) -> None:
    _malloc_trim()
    print(f"[mem {tag}] {_meminfo()}", flush=True)


def _strip_cpu_golden_refs(block) -> None:
    """Break the CPU-golden references kept by `enable_sparse_mlp` so the
    original per-expert weights can be GC'd before we ship to device.

    `enable_sparse_mlp` wires up a CPU fallback by (1) holding a hidden
    `_original_mlp` ref (stashed via `object.__setattr__` so it's not a
    submodule) and (2) registering a `original_experts: nn.ModuleList`
    of all 256 original Expert modules under the StackedExperts. Both
    pin ~13 GB of per-block CPU storage that streaming inference never
    needs — we only run on TT.

    This must run AFTER `enable_sparse_mlp(block, ...)` and BEFORE
    `block.to(device)` so the originals don't get shipped to TT.
    """
    ffn = getattr(block, "ffn", None)
    if ffn is None:
        return

    mlp = getattr(ffn, "mlp", ffn)  # A2aSparseMLPWithSharedExperts wraps .mlp

    # 1. _original_mlp stash (object.__setattr__, not in _modules)
    if hasattr(mlp, "_original_mlp"):
        object.__setattr__(mlp, "_original_mlp", None)

    # 2. original_experts ModuleList registered on StackedExperts
    experts = getattr(mlp, "experts", None)
    if experts is not None:
        modules_dict = getattr(experts, "_modules", {})
        if "original_experts" in modules_dict:
            del modules_dict["original_experts"]
        # Also break any direct attribute ref (defensive — _modules dict is
        # the authoritative store but nn.Module also caches via __setattr__).
        if hasattr(experts, "original_experts"):
            try:
                delattr(experts, "original_experts")
            except AttributeError:
                pass


# ----------------------------------------------------------------------------
# Per-block sharding spec. Mirrors `transformer_shard_spec` in
# test_deepseek_v4_full_e2e.py but operates on a single Block at a time so
# we can apply sharding immediately after shipping that block to the
# device.
# ----------------------------------------------------------------------------


def _block_shard_spec(block, mesh) -> Dict[torch.Tensor, Tuple]:
    """SPMD shard spec for one Block (post-sparse-MLP rewrite)."""
    specs: Dict[torch.Tensor, Tuple] = {}
    compound = ("_axis_0", "_axis_1")

    # Attention
    attn = block.attn
    specs[attn.wq_b.weight] = ("_axis_1", None)
    specs[attn.wo_a.weight] = ("_axis_1", None)
    specs[attn.wo_b.weight] = (None, "_axis_1")
    specs[attn.kv_cache] = ("_axis_0", None, None)
    if attn.compress_ratio:
        specs[attn.compressor.kv_cache] = ("_axis_0", None, None)
        specs[attn.compressor.kv_state] = ("_axis_0", None, None)
        specs[attn.compressor.score_state] = ("_axis_0", None, None)
        if getattr(attn, "indexer", None) is not None:
            specs[attn.indexer.wq_b.weight] = ("_axis_1", None)
            specs[attn.indexer.weights_proj.weight] = ("_axis_1", None)
            specs[attn.indexer.compressor.kv_cache] = ("_axis_0", None, None)
            specs[attn.indexer.compressor.kv_state] = ("_axis_0", None, None)
            specs[attn.indexer.compressor.score_state] = ("_axis_0", None, None)

    # MoE (post-swap A2aSparseMLPWithSharedExperts)
    a2a_with_shared = block.ffn
    mlp = a2a_with_shared.mlp
    specs[mlp.router.gate.weight] = (None, None)
    specs[mlp.experts.gate_proj] = (compound, None, None)
    specs[mlp.experts.up_proj] = (compound, None, None)
    specs[mlp.experts.down_proj] = (compound, None, None)

    shared = getattr(a2a_with_shared, "shared_experts", None)
    if shared is not None:
        specs[shared.w1.weight] = (None, None)
        specs[shared.w2.weight] = (None, None)
        specs[shared.w3.weight] = (None, None)

    return specs


def _top_level_shard_spec(model) -> Dict[torch.Tensor, Tuple]:
    """SPMD shard spec for embed / norm / head / hc_head_*."""
    return {
        model.embed.weight: (None, None),
        model.norm.weight: (None,),
        model.head.weight: (None, None),
        model.hc_head_fn: (None, None),
        model.hc_head_base: (None,),
        model.hc_head_scale: (None,),
    }


# ----------------------------------------------------------------------------
# Handle-only upload path. Bypasses `nn.Module.to(device)` to avoid
# torch_xla's `XLATensor::tensor_data` CPU shadow that would otherwise
# pin every parameter for the lifetime of the XLA tensor (the leak that
# made naive streaming useless — see OPEN_QUESTIONS.md #11).
#
# Uses `torch_xla._XLAC._xla_tensors_from_aten` which:
#   - calls `CreateTensorsData(tensors, shardings, devices)`
#   - in SPMD mode, physically shards the CPU tensor per local device
#     before upload (so peak host transient is bounded by full + shards)
#   - wraps the result in XLATensor::Create(handle) — sets only the data
#     handle, never sets `tensor_data`
# After the BufferFromHostBuffer transfer-complete lambda fires, the
# AtenSource refs drop and CPU storage is genuinely released.
# ----------------------------------------------------------------------------


def _upload_with_sharding(
    cpu_tensor: torch.Tensor,
    mesh,
    partition_spec,
    device,
) -> torch.Tensor:
    """Upload a CPU tensor to the XLA mesh with sharding pre-applied.

    `partition_spec=None` → upload via the implicit-replicated SPMD
    path (8 shallow at::Tensor refs sharing one storage; storage drops
    once all transfer-complete lambdas fire).
    """
    if partition_spec is None:
        return xm.send_cpu_data_to_device(
            [cpu_tensor], device, input_sharding=None,
        )[0]
    spec = xs.ShardingSpec(mesh, partition_spec)
    if not spec.can_apply(cpu_tensor):
        # Tensor shape doesn't match the partition_spec (e.g. 1-D tensor
        # with a 2-D partition_spec). Fall back to replicated.
        return xm.send_cpu_data_to_device(
            [cpu_tensor], device, input_sharding=None,
        )[0]
    return xm.send_cpu_data_to_device(
        [cpu_tensor], device, input_sharding=spec,
    )[0]


def _ship_module_handle_path(
    module: nn.Module,
    spec_by_id: Dict[int, Tuple],
    mesh,
    device,
    *,
    verbose: bool = False,
    tag: str = "",
) -> None:
    """Replace every Parameter and Buffer in `module` with an
    XLA-resident, handle-only copy uploaded via `_xla_tensors_from_aten`.

    `spec_by_id` is a `id(cpu_tensor) -> partition_spec` map; tensors
    not in the map upload as replicated.

    Mutates `module` in place. Drops references to the source CPU
    tensors; caller can then `gc.collect()` to actually release storage.
    """
    n_params = 0
    n_buffers = 0
    for sub in module.modules():
        # Parameters: walk and replace via `_parameters[name] = ...`
        # which is what nn.Module.__setattr__ does internally.
        for name, p in list(sub._parameters.items()):
            if p is None:
                continue
            partition_spec = spec_by_id.get(id(p))
            if p.device.type != "cpu":
                # Already migrated by a previous pass; skip.
                continue
            xla_t = _upload_with_sharding(
                p.data.detach(), mesh, partition_spec, device,
            )
            new_p = nn.Parameter(xla_t, requires_grad=False)
            sub._parameters[name] = new_p
            n_params += 1
        # Buffers (kv_cache, kv_state, etc.). `_buffers[name] = ...`
        # bypasses register_buffer's persistent flag handling but the
        # flag is preserved in `_non_persistent_buffers_set` separately.
        for name, b in list(sub._buffers.items()):
            if b is None:
                continue
            partition_spec = spec_by_id.get(id(b))
            if b.device.type != "cpu":
                continue
            xla_t = _upload_with_sharding(
                b.detach(), mesh, partition_spec, device,
            )
            sub._buffers[name] = xla_t
            n_buffers += 1
    if verbose:
        print(
            f"[stream] {tag} uploaded {n_params} params, {n_buffers} "
            "buffers via handle path",
            flush=True,
        )


# ----------------------------------------------------------------------------
# Streaming entry point.
# ----------------------------------------------------------------------------


def stream_load_transformer(
    args: mdo.ModelArgs,
    mesh,
    mesh_shape: Tuple[int, int],
    *,
    cluster_axis: int = 0,
    verbose: bool = True,
) -> mdo.Transformer:
    """Build a Transformer skeleton, then for each block load weights from
    HF, rewrite MoE, ship to TT, mark sharding, and free host storage
    before moving to the next block.

    Returns a fully-loaded device-resident model. Caller should still apply
    `apply_weight_dtype_overrides(model, …)` after this returns, BEFORE
    `torch.compile(model, backend="tt")`.
    """
    device = torch_xla.device()

    # Disable torch_xla's DeviceData CPU caching. By default torch_xla
    # retains the source CPU tensor for the lifetime of every XLA tensor
    # (so it can re-materialize on layout / sharding changes); for
    # streaming inference we don't need that, and the cache prevents the
    # host RAM streaming benefit entirely.
    if hasattr(torch_xla._XLAC, "_set_xla_enable_device_data_cache"):
        torch_xla._XLAC._set_xla_enable_device_data_cache(False)
        if verbose:
            print(
                "[stream] disabled torch_xla device-data cache "
                "(_set_xla_enable_device_data_cache(False))",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 1. Build empty CPU skeleton.
    # ------------------------------------------------------------------
    if verbose:
        _log_mem("pre-skeleton")
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)
    if verbose:
        _log_mem("post-skeleton")

    # ------------------------------------------------------------------
    # 2. Load top-level params (embed, norm, head, hc_head_*) and ship
    #    them to TT. Total ~3 GB so loading all at once is fine.
    # ------------------------------------------------------------------
    if verbose:
        print("[stream] loading top-level state dict ...", flush=True)
    # load_top_level_state_dict() returns head/norm/hc_head_* but NOT
    # embed.weight (which has its own loader function).
    embed_sd = weight_loader.load_embed_state_dict()
    model.embed.load_state_dict(embed_sd, strict=False)
    del embed_sd
    top_sd = weight_loader.load_top_level_state_dict()
    missing, unexpected = model.load_state_dict(top_sd, strict=False)
    if verbose:
        print(
            f"[stream] top-level: loaded {len(top_sd)} keys, "
            f"missing={len(missing)} unexpected={len(unexpected)}",
            flush=True,
        )
    del top_sd
    gc.collect()
    if verbose:
        _log_mem("post-top-level-load")

    # Build top-level sharding lookup BEFORE upload (params still on CPU).
    top_specs = _top_level_shard_spec(model)
    top_specs_by_id = {id(t): ps for t, ps in top_specs.items()}
    # Drop Tensor-keyed dict — keys hold strong refs to old CPU params.
    del top_specs

    # Ship via handle path: replaces each Parameter/Buffer in-place with
    # an XLA-resident, handle-only tensor. Sharding is applied during
    # upload so we don't need a separate mark_sharding call.
    _ship_module_handle_path(
        model.embed, top_specs_by_id, mesh, device,
        verbose=verbose, tag="top:embed",
    )
    _ship_module_handle_path(
        model.norm, top_specs_by_id, mesh, device,
        verbose=verbose, tag="top:norm",
    )
    _ship_module_handle_path(
        model.head, top_specs_by_id, mesh, device,
        verbose=verbose, tag="top:head",
    )
    # hc_head_* are direct nn.Parameter attrs on Transformer (not inside
    # a submodule). Walk Transformer's own _parameters dict.
    for pname in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        p = model._parameters.get(pname)
        if p is None or p.device.type != "cpu":
            continue
        partition_spec = top_specs_by_id.get(id(p))
        xla_t = _upload_with_sharding(
            p.data.detach(), mesh, partition_spec, device,
        )
        model._parameters[pname] = nn.Parameter(xla_t, requires_grad=False)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()
    if verbose:
        _log_mem("post-top-level-ship")

    # ------------------------------------------------------------------
    # 3. Stream the 43 blocks: load → rewrite-moe → ship → mark → free.
    # ------------------------------------------------------------------
    for layer_id in range(args.n_layers):
        if verbose:
            print(
                f"\n[stream] === block {layer_id}/{args.n_layers - 1} ===",
                flush=True,
            )
            _log_mem(f"pre-block-{layer_id}-load")

        # 3a. Load this block's HF weights.
        block_sd = weight_loader.load_block_state_dict(layer_id)
        # Strip `layers.{N}.` prefix if present so we can load on the block
        # directly. See OPEN_QUESTIONS.md item 3 — verify exact prefix.
        prefix = f"layers.{layer_id}."
        stripped = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in block_sd.items()
        }
        missing, unexpected = model.layers[layer_id].load_state_dict(
            stripped, strict=False,
        )
        if verbose:
            print(
                f"[stream] block {layer_id}: loaded {len(stripped)} keys, "
                f"missing={len(missing)} unexpected={len(unexpected)}",
                flush=True,
            )
        del block_sd, stripped
        gc.collect()
        if verbose:
            _log_mem(f"post-block-{layer_id}-load")

        # 3b. Rewrite this block's MoE (Block.ffn -> A2aSparseMLPWithSharedExperts)
        #     ON CPU. This is the StackedExperts construction step; it
        #     requires the per-expert tensors to be in CPU memory.
        enable_sparse_mlp(
            model.layers[layer_id],
            mesh=mesh_shape,
            cluster_axis=cluster_axis,
            config=args,
            verbose=False,
        )
        # Drop CPU-golden refs (~13 GB / block) before shipping. Without
        # this, host RAM grows linearly with layer count even though we
        # ship each block to device.
        _strip_cpu_golden_refs(model.layers[layer_id])
        gc.collect()
        if verbose:
            _log_mem(f"post-block-{layer_id}-sparse-rewrite")

        # 3c. Ship the (rewritten) block to TT via the handle path.
        #
        # Bypass `block.to(device)` entirely: that path stores the CPU
        # tensor in `XLATensor::data()->tensor_data` and never releases
        # it after upload (intentional shadow cache, see OPEN_QUESTIONS
        # blocker #11). Instead, we walk every Parameter and Buffer,
        # upload each one through `_xla_tensors_from_aten` with sharding
        # baked in, and replace it on its parent submodule. The sharding
        # spec is built BEFORE the upload while the block is still on
        # CPU (we look up by id() of the live CPU tensors).
        block = model.layers[layer_id]
        block_specs = _block_shard_spec(block, mesh)
        block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
        # Drop the Tensor-keyed dict immediately. Its keys hold strong
        # refs to the OLD CPU Parameters, which would keep the per-block
        # ~13 GB alive even after we replace `_parameters[name]` in
        # `_ship_module_handle_path`. block_specs_by_id is int-keyed so
        # safe to keep.
        del block_specs
        _ship_module_handle_path(
            block, block_specs_by_id, mesh, device,
            verbose=verbose, tag=f"block-{layer_id}",
        )
        torch_xla.sync(wait=True)
        # Force PJRT host->device transfer-complete callbacks to fire.
        # send_cpu_data_to_device uses kImmutableUntilTransferCompletes
        # semantics — the AtenSource (and its CPU at::Tensor) is held
        # alive by the BufferFromHostBuffer on-done lambda until the
        # transfer is acknowledged complete. Without this wait, the
        # callbacks only fire at process exit, defeating streaming.
        xm.wait_device_ops()
        gc.collect()
        if verbose:
            sample_devices = {
                str(p.device)
                for _, p in model.layers[layer_id].named_parameters()
            }
            print(
                f"[stream] block {layer_id}: param devices = {sample_devices}",
                flush=True,
            )
            _log_mem(f"post-block-{layer_id}-ship")
        else:
            # In non-verbose mode there's no _log_mem call to invoke
            # _malloc_trim, so do it here. Returns the per-block arena
            # tail to the OS; without it fragmentation accumulates and
            # host RSS climbs even after the patched torch_xla releases
            # the source tensors.
            _malloc_trim()

    if verbose:
        _log_mem("post-all-blocks")
        print("\n[stream] all blocks loaded and sharded.", flush=True)

    return model
