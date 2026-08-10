# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Selective HuggingFace weight loading for AI21-Jamba-Large-1.6 streaming.

Jamba is a hybrid Mamba + Attention + MoE architecture (``JambaForCausalLM``).
Checkpoints typically store MoE experts as per-expert
``feed_forward.experts.{E}.{gate,up,down}_proj``; modern ``transformers``
expects fused ``feed_forward.experts.gate_up_proj`` / ``down_proj`` (see
``conversion_mapping["jamba"]``). This loader remaps/fuses on the way in.

The HF repo is gated — set ``HF_TOKEN`` (or ``huggingface-cli login``) before
running.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoConfig, JambaConfig

MODEL_NAME = "ai21labs/AI21-Jamba-Large-1.6"
_VALID_MODEL_NAMES = (MODEL_NAME,)


def load_config(model_name: str = MODEL_NAME) -> JambaConfig:
    """Load JambaConfig (bf16). Forces slow Mamba path for TT (no CUDA kernels)."""
    assert model_name in _VALID_MODEL_NAMES
    config = AutoConfig.from_pretrained(model_name)
    config.torch_dtype = torch.bfloat16
    if hasattr(config, "dtype"):
        config.dtype = torch.bfloat16
    # Fast mamba-ssm / causal-conv1d kernels are CUDA-only; TT uses slow_forward.
    config.use_mamba_kernels = False
    config.use_cache = True
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
    """Fuse per-expert gate/up/down into ``feed_forward.experts.gate_up_proj/down_proj``."""
    moe_prefix = f"{layer_prefix}feed_forward."
    # Already fused in checkpoint?
    gu_key = f"{moe_prefix}experts.gate_up_proj"
    down_key = f"{moe_prefix}experts.down_proj"
    if gu_key in raw and down_key in raw:
        out = {
            "feed_forward.experts.gate_up_proj": _to_bf16(raw[gu_key]),
            "feed_forward.experts.down_proj": _to_bf16(raw[down_key]),
        }
        router_key = f"{moe_prefix}router.weight"
        if router_key in raw:
            out["feed_forward.router.weight"] = _to_bf16(raw[router_key])
        return out

    gates, ups, downs = [], [], []
    for e in range(num_experts):
        gates.append(raw[f"{moe_prefix}experts.{e}.gate_proj.weight"])
        ups.append(raw[f"{moe_prefix}experts.{e}.up_proj.weight"])
        downs.append(raw[f"{moe_prefix}experts.{e}.down_proj.weight"])

    gate_up = torch.cat([torch.stack(gates, dim=0), torch.stack(ups, dim=0)], dim=1)
    down = torch.stack(downs, dim=0)
    out = {
        "feed_forward.experts.gate_up_proj": _to_bf16(gate_up),
        "feed_forward.experts.down_proj": _to_bf16(down),
        "feed_forward.router.weight": _to_bf16(raw[f"{moe_prefix}router.weight"]),
    }
    return out


def _copy_non_expert_keys(
    raw: Dict[str, torch.Tensor], layer_prefix: str
) -> Dict[str, torch.Tensor]:
    """Copy layer tensors that are not per-expert Linear weights."""
    out: Dict[str, torch.Tensor] = {}
    expert_re = re.compile(rf"^{re.escape(layer_prefix)}feed_forward\.experts\.\d+\.")
    fused_re = re.compile(
        rf"^{re.escape(layer_prefix)}feed_forward\.experts\.(gate_up_proj|down_proj)$"
    )
    router_re = re.compile(rf"^{re.escape(layer_prefix)}feed_forward\.router\.")
    for full_key, tensor in raw.items():
        if not full_key.startswith(layer_prefix):
            continue
        if (
            expert_re.search(full_key)
            or fused_re.search(full_key)
            or router_re.search(full_key)
        ):
            continue
        rel = full_key[len(layer_prefix) :]
        out[rel] = _to_bf16(tensor)
    return out


def layer_has_moe(config: JambaConfig, layer_id: int) -> bool:
    """True when this layer uses ``JambaSparseMoeBlock`` (num_experts > 1)."""
    n = config.layers_num_experts[layer_id]
    return n > 1


def load_block_state_dict(
    model_name: str,
    layer_id: int,
    num_experts: Optional[int] = None,
    config: Optional[JambaConfig] = None,
) -> Dict[str, torch.Tensor]:
    """State dict matching one decoder layer (attention or mamba) with fused MoE."""
    prefix = f"model.layers.{layer_id}."
    raw = _load_raw_subset(model_name, [prefix])
    out = _copy_non_expert_keys(raw, prefix)

    # Dense MLP layers keep gate/up/down under feed_forward.* (already copied).
    # MoE layers need fuse + router.
    is_moe = False
    if config is not None:
        is_moe = layer_has_moe(config, layer_id)
    else:
        is_moe = any(
            re.search(rf"{re.escape(prefix)}feed_forward\.experts\.", k) for k in raw
        )

    if is_moe:
        if num_experts is None:
            expert_ids = {
                int(m.group(1))
                for k in raw
                if (
                    m := re.search(
                        rf"{re.escape(prefix)}feed_forward\.experts\.(\d+)\.", k
                    )
                )
            }
            if expert_ids:
                num_experts = max(expert_ids) + 1
            elif f"{prefix}feed_forward.experts.gate_up_proj" in raw:
                num_experts = raw[f"{prefix}feed_forward.experts.gate_up_proj"].shape[0]
            else:
                raise RuntimeError(f"No MoE experts found under {prefix}feed_forward")
        out.update(_fuse_experts(raw, prefix, num_experts))

    return out


def load_embed_state_dict(model_name: str = MODEL_NAME) -> Dict[str, torch.Tensor]:
    """``{"weight": embed_tokens}`` for ``model.embed_tokens``."""
    raw = _load_raw_subset(model_name, ["model.embed_tokens."])
    return {"weight": _to_bf16(raw["model.embed_tokens.weight"])}


def load_top_level_state_dict(model_name: str = MODEL_NAME) -> Dict[str, torch.Tensor]:
    """Final norm + lm_head (keys match ``JambaForCausalLM``)."""
    raw = _load_raw_subset(model_name, ["model.final_layernorm.", "lm_head."])
    return {
        "model.final_layernorm.weight": _to_bf16(raw["model.final_layernorm.weight"]),
        "lm_head.weight": _to_bf16(raw["lm_head.weight"]),
    }
