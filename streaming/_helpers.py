# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Internal helpers for the streaming runtime: skeleton builder, top-level
ship, buffer plumbing, host-memory diagnostic logging."""
from __future__ import annotations

import gc
import os
from typing import Dict, List, Tuple

import psutil
import torch
import torch_xla
from ttxla_tools.logging import logger
from torch import nn

from streaming.streaming_loader import _ship_module_handle_path, _upload_with_sharding
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)


def _malloc_trim() -> None:
    """Force glibc to return freed arenas to the OS so RSS tracks actual
    live allocations. No-op on platforms without `libc.malloc_trim`."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _check_no_unexpected(result, where: str) -> None:
    """Raise if `load_state_dict(..., strict=False)` returned any
    unexpected key — that signals a name-mapping bug that would otherwise
    leave the corresponding parameter uninitialized on device."""
    unexpected = getattr(result, "unexpected_keys", [])
    if unexpected:
        raise RuntimeError(
            f"{where}: load_state_dict found unexpected keys "
            f"(weights would not be applied): {sorted(unexpected)}"
        )


def _log(tag: str) -> None:
    """Trim then log RSS / system memory with a 38-char-padded tag."""
    _malloc_trim()
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    sys_used = psutil.virtual_memory().used / 1e9
    logger.info(f"[{tag:38s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")


def _build_skeleton(args):
    """Construct an empty (weight-less) `Transformer` in bf16 so subsequent
    `load_state_dict` calls write into bf16 buffers without an extra cast."""
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)


def _ship_top_level(
    model,
    mesh,
    device,
    *,
    top_level_shard_spec_fn,
    load_embed_state_dict,
    load_top_level_state_dict,
) -> None:
    """Load top-level state dicts (embed + norm + head + hc_head_*) and ship
    each to device with the partition spec the adapter provides."""
    embed_sd = load_embed_state_dict()
    _check_no_unexpected(
        model.embed.load_state_dict(embed_sd, strict=False),
        "embed",
    )
    del embed_sd
    top_sd = load_top_level_state_dict()
    _check_no_unexpected(
        model.load_state_dict(top_sd, strict=False),
        "top-level",
    )
    del top_sd
    gc.collect()
    top_specs = top_level_shard_spec_fn(model)
    top_specs_by_id = {id(t): ps for t, ps in top_specs.items()}
    del top_specs
    for sub_tag, sub in (
        ("top:embed", model.embed),
        ("top:norm", model.norm),
        ("top:head", model.head),
    ):
        _ship_module_handle_path(
            sub,
            top_specs_by_id,
            mesh,
            device,
            verbose=False,
            tag=sub_tag,
        )
    for pname in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        p = model._parameters.get(pname)
        if p is None or p.device.type != "cpu":
            continue
        partition_spec = top_specs_by_id.get(id(p))
        xla_t = _upload_with_sharding(
            p.data.detach(),
            mesh,
            partition_spec,
            device,
        )
        model._parameters[pname] = nn.Parameter(xla_t, requires_grad=False)
    torch_xla.sync(wait=True)
    gc.collect()


def _collect_buffer_paths(block) -> List[Tuple[nn.Module, str, str]]:
    """Walk `block` and return `(sub_module, buf_name, full_path)` for every
    registered buffer so callers can splice / capture them by name."""
    out = []
    for sub_path, sub in block.named_modules():
        for name, buf in list(sub._buffers.items()):
            if buf is None:
                continue
            full = f"{sub_path}.{name}" if sub_path else name
            out.append((sub, name, full))
    return out


def _splice_persistent_buffers(
    block,
    persistent_bufs: Dict[str, torch.Tensor],
) -> None:
    """In-place replace each buffer in `block` with the persistent device
    tensor at the same path, where present in `persistent_bufs`."""
    for sub, name, full in _collect_buffer_paths(block):
        if full in persistent_bufs:
            sub._buffers[name] = persistent_bufs[full]


def _ship_persistent_buffers_raw(
    block,
    mesh,
    device,
) -> Dict[str, torch.Tensor]:
    """Ship every CPU buffer in `block` to device with a batch-sharded
    spec (batch axis on `_axis_0`, other axes replicated). Returns the
    `{full_path: device_tensor}` map for later splicing."""
    out: Dict[str, torch.Tensor] = {}
    for sub, name, full in _collect_buffer_paths(block):
        b = sub._buffers[name]
        if b is None or b.device.type != "cpu":
            if b is not None:
                out[full] = b
            continue
        if b.dim() >= 3:
            partition_spec = ("_axis_0",) + (None,) * (b.dim() - 1)
        else:
            partition_spec = (None,) * b.dim()
        xla_t = _upload_with_sharding(
            b.detach(),
            mesh,
            partition_spec,
            device,
        )
        sub._buffers[name] = xla_t
        out[full] = xla_t
    torch_xla.sync(wait=True)
    return out
