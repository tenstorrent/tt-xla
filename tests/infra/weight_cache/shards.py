# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""HF safetensors-shard streaming helpers."""
import json
from collections.abc import Iterable, Mapping

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .paths import safe_open_hf


def open_hf_index(repo_id: str) -> dict[str, str]:
    """Download `model.safetensors.index.json` and return its `weight_map`.

    `weight_map` maps each checkpoint key to the shard filename that holds it.
    """
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with safe_open_hf(index_path) as f:
        index = json.load(f)
    return index["weight_map"]


def group_keys_by_shard(
    keys: Iterable[str], weight_map: Mapping[str, str]
) -> dict[str, list[str]]:
    """Invert `weight_map`: shard filename -> list of keys that live in it.

    Only the requested `keys` are included. Allows the loader to open each
    shard once and pull every needed tensor without re-opening.
    """
    shard_to_keys: dict[str, list[str]] = {}
    for k in keys:
        shard_to_keys.setdefault(weight_map[k], []).append(k)
    return shard_to_keys


def load_tensors_grouped(
    shard_to_keys: Mapping[str, list[str]], repo_id: str
) -> dict[str, torch.Tensor]:
    """Return `{ckpt_key: tensor}` for every requested key.

    Opens each shard once via `safe_open` for memory efficiency: large shards
    are mmap'd and only the requested tensors materialize.
    """
    out: dict[str, torch.Tensor] = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                out[key] = f.get_tensor(key)
    return out
