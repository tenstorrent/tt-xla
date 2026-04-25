# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Loads per-layer MoE/MLP weights from the deepseek-ai/DeepSeek-V4-Flash HF
# repo, dequantizing on the fly:
#
# - Routed experts (experts.{E}.w{1,2,3}.weight) are stored as FP4 (packed 2
#   values per byte as float4_e2m1fn_x2) with ue8m0 block scales [out, in/32].
#   Dequantization uses the FP4 lookup table from the repo's inference/convert.py.
# - Shared experts (shared_experts.w{1,2,3}.weight) are stored as FP8 e4m3fn
#   with ue8m0 block scales [out/128, in/128]. Dequantization is a 128x128
#   block multiply.
# - gate.weight / gate.bias ship as bf16 / fp32 and are loaded as-is.
#
# Only the shard(s) containing the requested layer are downloaded; subsequent
# runs hit the standard huggingface_hub cache.

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REPO_ID = "deepseek-ai/DeepSeek-V4-Flash"

# FP4 e2m1fn lookup: 4 bits -> float value. Bits 0-7 positive, 8-15 negative
# (bit 3 acts as sign). Copied verbatim from inference/convert.py.
_FP4_TABLE = torch.tensor(
    [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        3.0,
        4.0,
        6.0,
        0.0,
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -3.0,
        -4.0,
        -6.0,
    ],
    dtype=torch.float32,
)

_FP4_BLOCK = 32
_FP8_BLOCK = 128


def load_config_args():
    """Fetch inference/config.json and hydrate a model.ModelArgs with real values."""
    # The upstream model lives in third_party/tt_forge_models; the test file is
    # responsible for registering the `kernel` stub before any import touches it.
    from third_party.tt_forge_models.deepseek_v4.original_model import model

    path = hf_hub_download(repo_id=REPO_ID, filename="inference/config.json")
    with open(path) as f:
        raw = json.load(f)

    valid = {f.name for f in model.ModelArgs.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    # compress_ratios arrives as a list; the dataclass expects a tuple.
    if "compress_ratios" in kwargs:
        kwargs["compress_ratios"] = tuple(kwargs["compress_ratios"])
    # Force bf16 execution path: no FP4 experts, no FP8 non-expert linears.
    kwargs["dtype"] = "bf16"
    kwargs["expert_dtype"] = None
    kwargs["scale_fmt"] = None
    return model.ModelArgs(**kwargs)


def _dequant_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in/2] fp4/int8 packed + [out, in/32] ue8m0 -> [out, in] bf16."""
    # View as uint8 bytes regardless of whether safetensors returned int8 or fp4.
    byte_view = weight.contiguous().view(torch.uint8)
    out_dim, packed_in = byte_view.shape
    in_dim = packed_in * 2
    assert scale.shape == (
        out_dim,
        in_dim // _FP4_BLOCK,
    ), f"fp4 scale shape mismatch: weight={byte_view.shape}, scale={scale.shape}"

    table = _FP4_TABLE.to(byte_view.device)
    low = (byte_view & 0x0F).long()
    high = ((byte_view >> 4) & 0x0F).long()
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)  # [out, in]

    scale_f = scale.to(torch.float32).repeat_interleave(_FP4_BLOCK, dim=1)  # [out, in]
    return (vals * scale_f).to(torch.bfloat16)


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in] fp8_e4m3fn + [out/128, in/128] ue8m0 -> [out, in] bf16."""
    out_dim, in_dim = weight.shape
    assert (
        out_dim % _FP8_BLOCK == 0 and in_dim % _FP8_BLOCK == 0
    ), f"fp8 dims must be multiples of {_FP8_BLOCK}: got {weight.shape}"
    assert scale.shape == (
        out_dim // _FP8_BLOCK,
        in_dim // _FP8_BLOCK,
    ), f"fp8 scale shape mismatch: weight={weight.shape}, scale={scale.shape}"
    w = (
        weight.to(torch.float32)
        .unflatten(0, (-1, _FP8_BLOCK))
        .unflatten(-1, (-1, _FP8_BLOCK))
    )  # [bOut, 128, bIn, 128]
    s = scale.to(torch.float32)[:, None, :, None]  # [bOut, 1, bIn, 1]
    return (w * s).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)


def _find_shards_for_keys(
    weight_map: Dict[str, str], prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(prefixes)
    return sorted({shard for k, shard in weight_map.items() if k.startswith(prefixes)})


def _load_raw_subset(
    prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Download relevant shards and return tensors whose keys match any prefix.
    Keys are returned verbatim (with the original "layers.{L}.ffn." prefix)."""
    index_path = hf_hub_download(
        repo_id=REPO_ID, filename="model.safetensors.index.json"
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
        shard_path = hf_hub_download(repo_id=REPO_ID, filename=shard)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.startswith(prefix_tuple):
                    raw[key] = f.get_tensor(key)
    return raw


def _dequant_paired(
    raw: Dict[str, torch.Tensor], base_prefix: str
) -> Dict[str, torch.Tensor]:
    """Walk `raw` under `base_prefix`, combining `.weight`/`.scale` pairs into
    bf16 tensors keyed by the trimmed local name (base_prefix stripped).
    """
    out: Dict[str, torch.Tensor] = {}
    # First pass: partition into (local_name, has_scale, weight, scale).
    weights = {
        k: v
        for k, v in raw.items()
        if k.startswith(base_prefix) and k.endswith(".weight")
    }
    for wkey, w in weights.items():
        skey = wkey[: -len(".weight")] + ".scale"
        local = wkey[len(base_prefix) :]  # e.g. "experts.0.w1.weight"
        scale = raw.get(skey)
        if scale is None:
            out[local] = w.to(torch.bfloat16) if w.is_floating_point() else w
            continue
        # Heuristic: FP4 packed int8/fp4 has scale[-1] = weight.shape[-1] * 2 / 32.
        # FP8 has scale of shape [out/128, in/128].
        if scale.ndim == 2 and scale.shape == (
            w.shape[0],
            w.shape[1] * 2 // _FP4_BLOCK,
        ):
            out[local] = _dequant_fp4(w, scale)
        elif scale.ndim == 2 and scale.shape == (
            w.shape[0] // _FP8_BLOCK,
            w.shape[1] // _FP8_BLOCK,
        ):
            out[local] = _dequant_fp8(w, scale)
        else:
            raise RuntimeError(
                f"Unrecognized (weight, scale) shape pair for {wkey}: "
                f"w={tuple(w.shape)} s={tuple(scale.shape)}"
            )
    # Pass through non-weight/scale tensors (e.g. gate.bias) under base_prefix.
    for k, v in raw.items():
        if (
            not k.startswith(base_prefix)
            or k.endswith(".weight")
            or k.endswith(".scale")
        ):
            continue
        local = k[len(base_prefix) :]
        out[local] = v
    return out


def load_moe_state_dict(layer_id: int) -> Dict[str, torch.Tensor]:
    """State dict matching `model.MoE(layer_id, args).state_dict()` keys.

    Returns keys like: gate.weight, gate.bias, experts.{E}.w{1,2,3}.weight,
    shared_experts.w{1,2,3}.weight — all in bf16 (except gate.bias which
    remains in its checkpoint dtype, typically fp32).
    """
    base = f"layers.{layer_id}.ffn."
    raw = _load_raw_subset([base])
    return _dequant_paired(raw, base)


def load_expert_state_dict(layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
    """State dict matching `model.Expert(...).state_dict()` keys.

    Returns keys: w1.weight, w2.weight, w3.weight — bf16.
    """
    base = f"layers.{layer_id}.ffn.experts.{expert_id}."
    raw = _load_raw_subset([base])
    return _dequant_paired(raw, base)
