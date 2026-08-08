# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Selective HuggingFace weight loading for Mixtral-8x22B streaming bring-up.

Checkpoint shards still use the legacy per-expert layout
(``block_sparse_moe.experts.{E}.w{1,2,3}``). Modern ``transformers`` expects
fused ``mlp.experts.gate_up_proj`` / ``down_proj``, so this loader remaps and
fuses on the way in — same conversion as ``conversion_mapping["mixtral"]``.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, MixtralConfig

MODEL_NAME = "mistralai/Mixtral-8x22B-Instruct-v0.1"
_VALID_MODEL_NAMES = (MODEL_NAME,)


def load_config(model_name: str = MODEL_NAME) -> MixtralConfig:
    """Load MixtralConfig (bf16) for skeleton construction."""
    assert model_name in _VALID_MODEL_NAMES
    config = AutoConfig.from_pretrained(model_name)
    config.torch_dtype = torch.bfloat16
    if hasattr(config, "dtype"):
        config.dtype = torch.bfloat16
    return config


def _find_shards_for_keys(
    weight_map: Dict[str, str], prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(prefixes)
    return sorted({shard for k, shard in weight_map.items() if k.startswith(prefixes)})


def _load_raw_subset(
    model_name: str,
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Download relevant shards and return tensors whose keys match any prefix."""
    assert model_name in _VALID_MODEL_NAMES
    index_path = hf_hub_download(
        repo_id=model_name, filename="model.safetensors.index.json"
    )
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    shard_names = _find_shards_for_keys(weight_map, prefixes)
    if not shard_names:
        raise RuntimeError(f"No shards found for prefixes: {list(prefixes)}")

    raw: Dict[str, torch.Tensor] = {}
    prefix_tuple = tuple(prefixes)
    for shard in shard_names:
        shard_path = hf_hub_download(repo_id=model_name, filename=shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix_tuple):
                    raw[key] = f.get_tensor(key)
    return raw


def _to_bf16(t: torch.Tensor) -> torch.Tensor:
    return t.to(torch.bfloat16) if t.is_floating_point() else t


def _fuse_experts(
    raw: Dict[str, torch.Tensor], layer_prefix: str, num_experts: int
) -> Dict[str, torch.Tensor]:
    """Fuse per-expert w1/w2/w3 into gate_up_proj / down_proj under ``mlp.``."""
    moe_prefix = f"{layer_prefix}block_sparse_moe."
    w1s, w2s, w3s = [], [], []
    for e in range(num_experts):
        w1s.append(raw[f"{moe_prefix}experts.{e}.w1.weight"])
        w2s.append(raw[f"{moe_prefix}experts.{e}.w2.weight"])
        w3s.append(raw[f"{moe_prefix}experts.{e}.w3.weight"])

    # MergeModulelist(dim=0) then Concatenate(dim=1) for gate+up.
    gate_up = torch.cat([torch.stack(w1s, dim=0), torch.stack(w3s, dim=0)], dim=1)
    down = torch.stack(w2s, dim=0)
    return {
        "mlp.experts.gate_up_proj": _to_bf16(gate_up),
        "mlp.experts.down_proj": _to_bf16(down),
        "mlp.gate.weight": _to_bf16(raw[f"{moe_prefix}gate.weight"]),
    }


def load_block_state_dict(
    model_name: str, layer_id: int, num_experts: int | None = None
) -> Dict[str, torch.Tensor]:
    """State dict matching ``MixtralDecoderLayer.state_dict()`` keys (fused experts)."""
    prefix = f"model.layers.{layer_id}."
    raw = _load_raw_subset(model_name, [prefix])
    if num_experts is None:
        # Infer from checkpoint keys.
        expert_ids = {
            int(m.group(1))
            for k in raw
            if (
                m := re.search(
                    rf"{re.escape(prefix)}block_sparse_moe\.experts\.(\d+)\.", k
                )
            )
        }
        if not expert_ids:
            raise RuntimeError(f"No experts found under {prefix}block_sparse_moe")
        num_experts = max(expert_ids) + 1

    out = _fuse_experts(raw, prefix, num_experts)
    for src, dst in (
        ("self_attn.q_proj.weight", f"{prefix}self_attn.q_proj.weight"),
        ("self_attn.k_proj.weight", f"{prefix}self_attn.k_proj.weight"),
        ("self_attn.v_proj.weight", f"{prefix}self_attn.v_proj.weight"),
        ("self_attn.o_proj.weight", f"{prefix}self_attn.o_proj.weight"),
        ("input_layernorm.weight", f"{prefix}input_layernorm.weight"),
        ("post_attention_layernorm.weight", f"{prefix}post_attention_layernorm.weight"),
    ):
        out[src] = _to_bf16(raw[dst])
    return out


def load_embed_state_dict(model_name: str = MODEL_NAME) -> Dict[str, torch.Tensor]:
    """``{"weight": embed_tokens}`` for ``model.embed_tokens``."""
    raw = _load_raw_subset(model_name, ["model.embed_tokens."])
    return {"weight": _to_bf16(raw["model.embed_tokens.weight"])}


def load_top_level_state_dict(model_name: str = MODEL_NAME) -> Dict[str, torch.Tensor]:
    """Final norm + lm_head (keys match ``MixtralForCausalLM``)."""
    raw = _load_raw_subset(model_name, ["model.norm.", "lm_head."])
    return {
        "model.norm.weight": _to_bf16(raw["model.norm.weight"]),
        "lm_head.weight": _to_bf16(raw["lm_head.weight"]),
    }
