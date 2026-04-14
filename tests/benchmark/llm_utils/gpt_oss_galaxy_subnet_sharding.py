# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Apply Galaxy benchmark shard specs to deep-copied MoE / post-attn-norm modules."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh


def map_benchmark_shard_specs_to_module_copy(
    orig_module: torch.nn.Module,
    copy_module: torch.nn.Module,
    full_shard_specs: dict,
) -> dict:
    """Map ``full_shard_specs`` keys from ``orig_module`` parameters to ``copy_module``."""

    orig_by_name = dict(orig_module.named_parameters())
    out = {}
    for name, p_copy in copy_module.named_parameters():
        p_orig = orig_by_name.get(name)
        if p_orig is not None and p_orig in full_shard_specs:
            out[p_copy] = full_shard_specs[p_orig]
    return out


def mark_gpt_oss_galaxy_copied_subnet_weights(
    model: torch.nn.Module,
    model_loader,
    layer_idx: int,
    mesh: Mesh,
    mlp_copy: torch.nn.Module,
    post_norm_copy: torch.nn.Module | None,
    shard_spec_fn: Callable,
) -> None:
    """Mark sharding on TT MoE / optional post-attn RMSNorm copies (Galaxy 20B benchmark layout)."""

    full = shard_spec_fn(model_loader, model)
    layer = model.model.layers[layer_idx]
    for tensor, spec in map_benchmark_shard_specs_to_module_copy(
        layer.mlp, mlp_copy, full
    ).items():
        xs.mark_sharding(tensor, mesh, spec)
    if post_norm_copy is not None:
        for tensor, spec in map_benchmark_shard_specs_to_module_copy(
            layer.post_attention_layernorm, post_norm_copy, full
        ).items():
            xs.mark_sharding(tensor, mesh, spec)
