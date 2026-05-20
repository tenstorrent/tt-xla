# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cache build orchestrator.

`build_cache(spec)` materializes one cache directory; `ensure_cache(spec)`
recursively materializes the chain (`spec.next_stage` first, then `spec`) if
either side is missing.

Two source modes, distinguished by which transform field on the spec is set:

- HF-source: iterate `spec.iter_groups(weight_map)`, stream the relevant HF
  shards via `load_tensors_grouped`, call `spec.transform_group`, save one
  chunk per group.
- Cache-source: iterate `.safetensors` files in `spec.next_stage.cache_dir`,
  load each chunk, call `spec.transform_chunk(chunk, chunk_name)`, save under
  the same filename in `spec.cache_dir`.
"""
import time
from pathlib import Path

from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from .paths import has_cache
from .shards import group_keys_by_shard, load_tensors_grouped, open_hf_index
from .spec import WeightCacheSpec


def ensure_cache(spec: WeightCacheSpec) -> Path:
    """Recursively build `spec` and any `next_stage` that isn't on disk yet.

    Logs whether each stage was reused or built so callers can tell at a
    glance whether a long build is about to start. Returns the final cache
    directory (`spec.cache_dir`).
    """
    if spec.next_stage is not None:
        ensure_cache(spec.next_stage)
    if has_cache(spec.cache_dir):
        print(f"[cache] using existing cache at {spec.cache_dir}", flush=True)
    else:
        print(
            f"[cache] building cache at {spec.cache_dir} (from {spec.repo_id})",
            flush=True,
        )
        build_cache(spec)
    return spec.cache_dir


def build_cache(spec: WeightCacheSpec) -> None:
    """Build one cache directory from scratch.

    Caller is responsible for choosing the right source mode by populating
    either (iter_groups + transform_group) or transform_chunk on the spec.
    When the cache already exists, this is a no-op (with a log line) so the
    CLI builders can be safely re-run.
    """
    if has_cache(spec.cache_dir):
        print(f"[cache] already exists at {spec.cache_dir}, skipping.", flush=True)
        return

    spec.cache_dir.mkdir(parents=True, exist_ok=True)

    if spec.transform_chunk is not None:
        _build_from_cache(spec)
    else:
        if spec.iter_groups is None or spec.transform_group is None:
            raise ValueError(
                "WeightCacheSpec needs either (iter_groups + transform_group) "
                "or transform_chunk"
            )
        _build_from_hf(spec)


def _build_from_hf(spec: WeightCacheSpec) -> None:
    """HF-source build: stream shards per group, transform, save."""
    t_total = time.perf_counter()
    weight_map = open_hf_index(spec.repo_id)

    total_bytes = 0
    n_chunks = 0
    for group in spec.iter_groups(weight_map):
        t_group = time.perf_counter()
        all_keys = list(group.ckpt_keys) + list(group.aux_keys)
        shard_to_keys = group_keys_by_shard(all_keys, weight_map)
        raw = load_tensors_grouped(shard_to_keys, spec.repo_id)
        out = spec.transform_group(raw, group)
        del raw

        chunk_path = spec.cache_dir / f"{group.name}.safetensors"
        safetensors_save_file(out, str(chunk_path))
        sz = chunk_path.stat().st_size
        total_bytes += sz
        n_chunks += 1
        print(
            f"  {group.name}: {len(out)} keys, {sz / 1e9:.2f} GB, "
            f"{time.perf_counter() - t_group:.1f}s",
            flush=True,
        )
        del out

    print(
        f"[cache] done: {n_chunks} chunks, {total_bytes / 1e9:.2f} GB total, "
        f"{time.perf_counter() - t_total:.1f}s -> {spec.cache_dir}",
        flush=True,
    )


def _build_from_cache(spec: WeightCacheSpec) -> None:
    """Cache-source build: transform each chunk in next_stage.cache_dir."""
    if spec.next_stage is None:
        raise ValueError("transform_chunk requires next_stage to provide source chunks")
    source_dir = spec.next_stage.cache_dir

    t_total = time.perf_counter()
    total_bytes = 0
    for fname in sorted(p.name for p in source_dir.iterdir()):
        if not fname.endswith(".safetensors"):
            continue
        t_chunk = time.perf_counter()
        chunk_name = fname[: -len(".safetensors")]
        raw = safetensors_load_file(str(source_dir / fname))
        out = spec.transform_chunk(raw, chunk_name)
        del raw

        chunk_path = spec.cache_dir / fname
        safetensors_save_file(out, str(chunk_path))
        sz = chunk_path.stat().st_size
        total_bytes += sz
        print(
            f"  {fname}: {sz / 1e9:.2f} GB, " f"{time.perf_counter() - t_chunk:.1f}s",
            flush=True,
        )
        del out

    print(
        f"[cache] done: {total_bytes / 1e9:.2f} GB total, "
        f"{time.perf_counter() - t_total:.1f}s -> {spec.cache_dir}",
        flush=True,
    )
