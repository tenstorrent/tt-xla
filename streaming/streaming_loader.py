# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Handle-only XLA upload + post-sparse-MLP CPU ref strip for the
streaming runtime."""

from __future__ import annotations

from typing import Dict

import torch
import torch_xla.distributed.spmd as xs
from torch import nn
from ttxla_tools.logging import logger


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


def _upload_with_sharding(
    cpu_tensor: torch.Tensor,
    mesh,
    partition_spec,
    device,
) -> torch.Tensor:
    """Upload a CPU tensor to the XLA mesh and annotate it with the
    requested sharding via the standard `.to(device)` + `mark_sharding`
    path. `partition_spec=None` leaves the tensor replicated."""
    xla_t = cpu_tensor.to(device)
    if partition_spec is None:
        return xla_t
    if len(partition_spec) != cpu_tensor.dim():
        raise ValueError(
            f"partition_spec {partition_spec!r} rank does not match tensor "
            f"with shape {tuple(cpu_tensor.shape)}"
        )
    xs.mark_sharding(xla_t, mesh, partition_spec)
    return xla_t


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
                p.data.detach(),
                mesh,
                partition_spec,
                device,
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
                b.detach(),
                mesh,
                partition_spec,
                device,
            )
            sub._buffers[name] = xla_t
            n_buffers += 1
    if verbose:
        logger.info(
            f"[stream] {tag} uploaded {n_params} params, {n_buffers} "
            "buffers via handle path",
        )
