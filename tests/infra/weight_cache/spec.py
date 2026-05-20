# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Dataclasses describing how to build a weight cache.

A `WeightCacheSpec` is a pure-data description of one cache directory: where to
write it, where the raw tensors come from, and how to transform them. The
orchestrator in `builder.py` consumes the spec; per-model logic (rename tables,
dequant, expert stacking) lives in the closures the spec carries.

Two source modes are supported, chosen by which transform field is populated:

- **HF source** (set `iter_groups` + `transform_group`): build by iterating
  output chunks. For each `GroupDef`, the orchestrator streams the relevant HF
  shards, hands the raw `{ckpt_key: tensor}` dict to `transform_group`, and
  saves the returned `{model_key: tensor}` dict as one chunk.
- **Cache source** (set `transform_chunk`): build by iterating chunk files in
  `next_stage.cache_dir`. Each chunk is loaded, passed to `transform_chunk`
  with its name, and written to `cache_dir` under the same filename. Used for
  DeepSeek's post-sparse expert-stacking stage.

`next_stage` declares a dependency: the orchestrator ensures it exists before
building the current spec. For cache-source mode, `next_stage` is also the
data source.

These dataclasses currently live in tt-xla. They are intentionally I/O-free so
they can move to `third_party/tt_forge_models/utils/` later without dragging
the orchestrator (which talks to HF and safetensors) into the submodule.
"""
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class GroupDef:
    """One output chunk in an HF-source build.

    The orchestrator uses `ckpt_keys` and `aux_keys` to figure out which HF
    shards to fetch. `metadata` is opaque to the orchestrator and lets the
    per-model `transform_group` closure carry state across the iter/transform
    boundary (e.g. a `ckpt_key -> model_key` rename dict).
    """

    name: str  # e.g. "shared", "layer_0007" -> becomes "{name}.safetensors"
    ckpt_keys: list[str] = field(default_factory=list)
    aux_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class WeightCacheSpec:
    """How to build one chunked safetensors cache directory.

    Exactly one of (iter_groups + transform_group) or transform_chunk should be
    set — see the module docstring for the two source modes.
    """

    repo_id: str  # HF repo this cache was derived from (for context/logging)
    cache_dir: Path  # destination directory; one .safetensors file per group/chunk

    # HF-source mode:
    iter_groups: Callable[[Mapping[str, str]], Iterator[GroupDef]] | None = None
    transform_group: (
        Callable[[Mapping[str, torch.Tensor], GroupDef], dict[str, torch.Tensor]] | None
    ) = None

    # Cache-source mode:
    transform_chunk: (
        Callable[[dict[str, torch.Tensor], str], dict[str, torch.Tensor]] | None
    ) = None

    # Dependency chain: ensure this is built first (and, for cache-source mode,
    # also the data source).
    next_stage: "WeightCacheSpec | None" = None
