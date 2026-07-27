# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-block streaming weight loader for DeepSeek-V3.1.

Reads the BF16 mirror of the checkpoint one transformer block at a time so peak
host RAM stays bounded to roughly a single block's worth of weights. This is the
V3.1 analog of ``tests/torch/models/deepseek_v4/weight_loader.py`` -- but the
mirror ships plain bf16 tensors, so there is no fp4/fp8 dequant here.

The getters return state dicts keyed to match the *module* they load into:
  * ``load_block_state_dict(repo, i)`` -> keys rooted at a single
    ``DeepseekV3DecoderLayer`` (``self_attn.q_a_proj.weight``, ``mlp.gate.weight``,
    ``mlp.experts.0.gate_proj.weight``, ``mlp.shared_experts...``).
  * ``load_top_level_state_dict(repo)`` -> the full-model keys for the
    non-layer params (``model.embed_tokens.weight``, ``model.norm.weight``,
    ``lm_head.weight``); load with ``model.load_state_dict(..., strict=False)``.

Load every result with ``strict=False`` -- non-persistent buffers (rotary
inv_freq / cos_cached / sin_cached) are not in the checkpoint.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Multi-token-prediction / next-N keys are not part of the decoder stack.
_DROP_PREFIXES: Tuple[str, ...] = ("mtp.",)
_DROP_SUBSTRINGS: Tuple[str, ...] = (".mtp.",)


def _drop_key(key: str) -> bool:
    return any(key.startswith(p) for p in _DROP_PREFIXES) or any(
        s in key for s in _DROP_SUBSTRINGS
    )


def _load_index(model_name: str) -> Optional[Dict[str, str]]:
    """Return the checkpoint's ``weight_map`` (key -> shard filename), or ``None``
    for a single-shard repo that has no ``model.safetensors.index.json``."""
    try:
        index_path = hf_hub_download(
            repo_id=model_name, filename="model.safetensors.index.json"
        )
    except Exception:
        return None
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def _shards_for_prefixes(
    weight_map: Optional[Dict[str, str]], prefixes: Tuple[str, ...]
) -> List[str]:
    """Sorted unique shard filenames holding any key that starts with a prefix.
    A ``None`` weight_map (single-shard repo) resolves to ``model.safetensors``."""
    if weight_map is None:
        return ["model.safetensors"]
    shards = {
        shard
        for key, shard in weight_map.items()
        if key.startswith(prefixes) and not _drop_key(key)
    }
    return sorted(shards)


def _load_keys_with_prefixes(
    model_name: str, prefixes: Iterable[str]
) -> Dict[str, torch.Tensor]:
    """Download only the shards holding keys matching ``prefixes`` and read just
    those keys (bounded host RAM). Keys are returned verbatim."""
    prefixes = tuple(prefixes)
    weight_map = _load_index(model_name)
    shard_names = _shards_for_prefixes(weight_map, prefixes)
    if not shard_names:
        raise RuntimeError(f"No shards found for prefixes: {list(prefixes)}")

    out: Dict[str, torch.Tensor] = {}
    for shard in shard_names:
        shard_path = hf_hub_download(repo_id=model_name, filename=shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefixes) and not _drop_key(key):
                    out[key] = f.get_tensor(key)
    return out


def _to_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16) if t.is_floating_point() else t


def load_block_state_dict(model_name: str, layer_id: int) -> Dict[str, torch.Tensor]:
    """State dict for one ``DeepseekV3DecoderLayer`` (keys stripped of the
    ``model.layers.{layer_id}.`` prefix)."""
    prefix = f"model.layers.{layer_id}."
    raw = _load_keys_with_prefixes(model_name, [prefix])
    if not raw:
        raise RuntimeError(f"No weights found for layer {layer_id} (prefix {prefix!r})")
    return {key[len(prefix) :]: _to_bf16(v) for key, v in raw.items()}


def load_top_level_state_dict(model_name: str) -> Dict[str, torch.Tensor]:
    """State dict for the non-layer params, keyed as the full model expects:
    ``model.embed_tokens.weight``, ``model.norm.weight``, ``lm_head.weight``."""
    raw = _load_keys_with_prefixes(
        model_name, ["model.embed_tokens.", "model.norm.", "lm_head."]
    )
    return {key: _to_bf16(v) for key, v in raw.items()}
