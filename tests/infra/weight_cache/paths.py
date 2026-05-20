# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Cache-directory layout and HF-cache safe-open helper.

Cache root: `$HF_HOME/tt_xla_weight_cache/` (default `~/.cache/huggingface/...`).
Per-variant subdirectory: `{slug}_{n_layers}layers_{variant}`, where `slug` is
the HF repo id with `/` replaced by `--`, and `variant` is currently one of:

- `"bf16"`: base BF16 cache (after per-model rename, with FP8 sources dequanted)
- `"stacked"`: post-sparse stacked-experts cache for `enable_sparse_mlp`

Add new variants by passing a new string; the orchestrator treats the value as
opaque.
"""
import os
from pathlib import Path
from typing import IO


def _hf_home() -> Path:
    """Resolve `$HF_HOME` (default `~/.cache/huggingface`) to an absolute path."""
    return Path(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ).resolve()


def safe_open_hf(path: str | os.PathLike) -> IO:
    """Open `path` only if it resolves under `$HF_HOME`.

    Used when reading HF-downloaded files (e.g. `model.safetensors.index.json`)
    to guard against following symlinks out of the cache.
    """
    real = os.path.realpath(path)
    base = str(_hf_home())
    if not real.startswith(base + os.sep):
        raise ValueError(f"Path outside HF cache: {path}")
    return open(real)


def cache_dir_for(repo_id: str, n_layers: int, variant: str = "bf16") -> Path:
    """Canonical cache path for a (repo, n_layers, variant) triple.

    Returns `$HF_HOME/tt_xla_weight_cache/{slug}_{n_layers}layers_{variant}`
    where `slug = repo_id.replace("/", "--")`. The variant string is appended
    verbatim, so callers can pass any variant name the loader supports.
    """
    repo_slug = repo_id.replace("/", "--")
    return (
        _hf_home() / "tt_xla_weight_cache" / f"{repo_slug}_{n_layers}layers_{variant}"
    )


def has_cache(cache_dir: os.PathLike) -> bool:
    """True iff `cache_dir` exists and contains at least one .safetensors file."""
    p = Path(cache_dir)
    return p.is_dir() and any(f.suffix == ".safetensors" for f in p.iterdir())
