# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Shared weight-cache infrastructure for torch model tests.

Per-model rename tables, group iterators, expert-stacking layout, and any
post-load fixes live in each model's loader / build script — they describe the
*per-model* transform. This package provides the orchestrator: cache-directory
layout, HF shard streaming, FP8 dequant, the build loop, and a load helper for
materializing a meta-initialized model from a cache directory.

Typical usage:

    from tests.infra.weight_cache import (
        WeightCacheSpec, GroupDef, cache_dir_for,
        ensure_cache, load_cache_into,
        open_hf_index, group_keys_by_shard, load_tensors_grouped,
        fp8_blockwise_dequant, maybe_dequant,
    )

    spec = WeightCacheSpec(
        repo_id="some-org/some-model",
        cache_dir=cache_dir_for("some-org/some-model", n_layers, variant="bf16"),
        iter_groups=my_iter_groups,
        transform_group=my_transform_group,
    )
    ensure_cache(spec)
    missing, unexpected = load_cache_into(model, spec.cache_dir)
"""
from .builder import build_cache, ensure_cache
from .dequant import fp8_blockwise_dequant, is_fp8, maybe_dequant
from .loading import load_cache_into
from .paths import cache_dir_for, has_cache, safe_open_hf
from .shards import group_keys_by_shard, load_tensors_grouped, open_hf_index
from .spec import GroupDef, WeightCacheSpec

__all__ = [
    "GroupDef",
    "WeightCacheSpec",
    "build_cache",
    "ensure_cache",
    "cache_dir_for",
    "has_cache",
    "safe_open_hf",
    "open_hf_index",
    "group_keys_by_shard",
    "load_tensors_grouped",
    "fp8_blockwise_dequant",
    "is_fp8",
    "maybe_dequant",
    "load_cache_into",
]
