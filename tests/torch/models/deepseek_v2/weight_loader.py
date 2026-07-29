# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Per-layer HuggingFace weight loading for DeepSeek-V2-Chat streaming e2e.

DeepSeek-V2-Chat is stored as BF16 safetensors (~471 GB across 55 shards). The
streaming test never materializes the full state dict: each call loads only the
keys for one decoder layer (or the small top-level embed / norm / lm_head set).
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Chat"


def load_config(model_name: str = MODEL_NAME):
    """Load HF config with eager attention (required for torch.compile / TT)."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config._attn_implementation = "eager"
    config.torch_dtype = torch.bfloat16
    return config


def _load_weight_map(model_name: str) -> Dict[str, str]:
    index_path = hf_hub_download(
        repo_id=model_name, filename="model.safetensors.index.json"
    )
    with open(index_path) as f:
        return json.load(f)["weight_map"]


def _find_shards_for_prefixes(
    weight_map: Dict[str, str], prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(prefixes)
    return sorted(
        {shard for k, shard in weight_map.items() if k.startswith(prefixes)}
    )


def _load_raw_subset(
    model_name: str, prefixes: Iterable[str]
) -> Dict[str, torch.Tensor]:
    """Download relevant shards and return tensors whose keys match any prefix."""
    weight_map = _load_weight_map(model_name)
    shard_names = _find_shards_for_prefixes(weight_map, prefixes)
    if not shard_names:
        raise RuntimeError(f"No shards found for prefixes: {list(prefixes)}")

    raw: Dict[str, torch.Tensor] = {}
    prefix_tuple = tuple(prefixes)
    for shard in shard_names:
        shard_path = hf_hub_download(repo_id=model_name, filename=shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix_tuple):
                    t = f.get_tensor(key)
                    raw[key] = t.to(torch.bfloat16) if t.is_floating_point() else t
    return raw


def load_block_state_dict(
    model_name: str, layer_id: int
) -> Dict[str, torch.Tensor]:
    """State dict for `model.model.layers[layer_id]` (keys rooted at the block).

    Checkpoint keys look like ``model.layers.{L}.self_attn....``; returned keys
    strip that prefix so they match ``DeepseekV2DecoderLayer.state_dict()``.
    """
    prefix = f"model.layers.{layer_id}."
    raw = _load_raw_subset(model_name, [prefix])
    return {k[len(prefix) :]: v for k, v in raw.items()}


def load_embed_state_dict(model_name: str) -> Dict[str, torch.Tensor]:
    """Embedding weights for `model.model.embed_tokens`."""
    raw = _load_raw_subset(model_name, ["model.embed_tokens."])
    return {
        k[len("model.embed_tokens.") :]: v
        for k, v in raw.items()
        if k.startswith("model.embed_tokens.")
    }


def load_top_level_state_dict(model_name: str) -> Dict[str, torch.Tensor]:
    """Final norm + lm_head (keys match the CausalLM module namespace)."""
    raw = _load_raw_subset(model_name, ["model.norm.", "lm_head."])
    out: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        out[k] = v
    return out
