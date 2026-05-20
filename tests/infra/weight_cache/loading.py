# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Load a built cache into a model.

Reads every `.safetensors` file in `cache_dir` via mmap (the safetensors
default), merges into one state_dict, and calls `model.load_state_dict(...,
assign=True)`. `assign=True` swaps tensors in by reference, which is what
meta-initialized models need to become materialized.
"""
import os

import torch
from safetensors.torch import load_file as safetensors_load_file


def load_cache_into(
    model: torch.nn.Module,
    cache_dir: os.PathLike,
    *,
    strict: bool = False,
) -> tuple[list[str], list[str]]:
    """Load every chunk in `cache_dir` into `model`.

    Returns the `(missing, unexpected)` tuple from `load_state_dict`. With
    `strict=False` (default) the caller can choose to log or assert on those
    lists separately — useful for sparse-MoE models where some buffers are
    materialized post-load.
    """
    cache_dir = str(cache_dir)
    state_dict: dict[str, torch.Tensor] = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".safetensors"):
            continue
        state_dict.update(safetensors_load_file(os.path.join(cache_dir, fname)))

    missing, unexpected = model.load_state_dict(state_dict, strict=strict, assign=True)
    return list(missing), list(unexpected)
