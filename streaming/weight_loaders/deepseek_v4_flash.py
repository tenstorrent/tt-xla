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
import os
import time
from typing import Dict, Iterable, List

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from ttxla_tools.logging import logger

# Per-layer timing breakdown. Accumulates across calls; load_block_state_dict
# prints + resets at end. Diagnostic only — set DSV4_TIMING=0 to silence.
_TIMERS: Dict[str, float] = {
    "io": 0.0,  # safetensors disk I/O (get_tensor wall-clock)
    "fp4_time": 0.0,  # _dequant_fp4 wall-clock
    "fp4_count": 0,  # number of fp4 tensors dequantized
    "fp8_time": 0.0,  # _dequant_fp8 wall-clock
    "fp8_count": 0,  # number of fp8 tensors dequantized
}
_TIMING_ENABLED = os.environ.get("DSV4_TIMING", "1") == "1"
# A/B switch: DSV4_FP4_BF16=0 falls back to the fp32 dequant path (multiply
# in fp32, then cast to bf16). Default 1 = bf16 native path.
_FP4_BF16 = os.environ.get("DSV4_FP4_BF16", "1") == "1"


def _reset_timers():
    for k in _TIMERS:
        _TIMERS[k] = 0.0 if isinstance(_TIMERS[k], float) else 0


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
    from third_party.tt_forge_models.deepseek_v4.modified_model import model

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
    """[out, in/2] fp4/int8 packed + [out, in/32] ue8m0 -> [out, in] bf16.

    Multiplies in bf16 directly: the 16-entry FP4 LUT (max |val|=6) and ue8m0
    scales (powers of 2) both round-trip exactly through bf16, so skipping the
    fp32 intermediate halves memory traffic with no accuracy loss.
    """
    t0 = time.time() if _TIMING_ENABLED else 0.0
    byte_view = weight.contiguous().view(torch.uint8)
    out_dim, packed_in = byte_view.shape
    in_dim = packed_in * 2
    expected = (out_dim, in_dim // _FP4_BLOCK)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"fp4 scale shape mismatch: weight={byte_view.shape}, "
            f"scale={scale.shape}, expected={expected}"
        )

    if _FP4_BF16:
        table = _FP4_TABLE.to(byte_view.device, dtype=torch.bfloat16)
    else:
        table = _FP4_TABLE.to(byte_view.device)
    low = (byte_view & 0x0F).long()
    high = ((byte_view >> 4) & 0x0F).long()
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)

    if _FP4_BF16:
        scale_x = scale.to(torch.bfloat16).repeat_interleave(_FP4_BLOCK, dim=1)
        out = vals * scale_x
    else:
        scale_x = scale.to(torch.float32).repeat_interleave(_FP4_BLOCK, dim=1)
        out = (vals * scale_x).to(torch.bfloat16)
    if _TIMING_ENABLED:
        _TIMERS["fp4_time"] += time.time() - t0
        _TIMERS["fp4_count"] += 1
    return out


def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in] fp8_e4m3fn + [out/128, in/128] ue8m0 -> [out, in] bf16."""
    t0 = time.time() if _TIMING_ENABLED else 0.0
    out_dim, in_dim = weight.shape
    if out_dim % _FP8_BLOCK != 0 or in_dim % _FP8_BLOCK != 0:
        raise ValueError(
            f"fp8 dims must be multiples of {_FP8_BLOCK}: got {weight.shape}"
        )
    expected = (out_dim // _FP8_BLOCK, in_dim // _FP8_BLOCK)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"fp8 scale shape mismatch: weight={weight.shape}, "
            f"scale={scale.shape}, expected={expected}"
        )
    w = (
        weight.to(torch.float32)
        .unflatten(0, (-1, _FP8_BLOCK))
        .unflatten(-1, (-1, _FP8_BLOCK))
    )  # [bOut, 128, bIn, 128]
    s = scale.to(torch.float32)[:, None, :, None]  # [bOut, 1, bIn, 1]
    out = (w * s).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)
    if _TIMING_ENABLED:
        _TIMERS["fp8_time"] += time.time() - t0
        _TIMERS["fp8_count"] += 1
    return out


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
                    t0 = time.time() if _TIMING_ENABLED else 0.0
                    raw[key] = f.get_tensor(key)
                    if _TIMING_ENABLED:
                        _TIMERS["io"] += time.time() - t0
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


def load_block_state_dict(layer_id: int) -> Dict[str, torch.Tensor]:
    """State dict matching `model.Block(layer_id, args).state_dict()` keys.

    Pulls the full Block: attention (incl. compressor/indexer), attn_norm,
    ffn (MoE — gate, experts, shared), ffn_norm, hc_attn_*, hc_ffn_*.

    Heavy: hash-routed layers contain 256 routed experts × 3 weight matrices
    (fp4-packed in storage; ~3GB on disk per layer, ~12GB after bf16 dequant
    in memory).
    """
    if _TIMING_ENABLED:
        _reset_timers()
        t_total = time.time()
    base = f"layers.{layer_id}."
    raw = _load_raw_subset([base])
    out = _dequant_paired(raw, base)
    if _TIMING_ENABLED:
        elapsed = time.time() - t_total
        io = _TIMERS["io"]
        fp4_t = _TIMERS["fp4_time"]
        fp4_n = _TIMERS["fp4_count"]
        fp8_t = _TIMERS["fp8_time"]
        fp8_n = _TIMERS["fp8_count"]
        other = elapsed - io - fp4_t - fp8_t
        logger.info(
            f"[wl l{layer_id}] total={elapsed:.2f}s "
            f"io={io:.2f}s "
            f"fp4={fp4_t:.2f}s/{fp4_n} "
            f"fp8={fp8_t:.2f}s/{fp8_n} "
            f"other={other:.2f}s",
        )
    return out


def load_embed_state_dict() -> Dict[str, torch.Tensor]:
    """Top-level embedding weights for `Transformer.embed`.

    Returns: {"weight": tensor[vocab_size, dim] bf16}.
    """
    raw = _load_raw_subset(["embed."])
    return _dequant_paired(raw, "embed.")


def load_top_level_state_dict() -> Dict[str, torch.Tensor]:
    """Top-level Transformer params that don't live under a layer.

    Includes hc_head_fn / hc_head_base / hc_head_scale (the head-side
    Hyper-Connections params that reduce hc_mult copies back to one), the
    final RMSNorm `norm.weight`, and the parallel head's `head.weight`.
    Keys are returned verbatim with their top-level dotted names.
    """
    raw = _load_raw_subset(
        [
            "hc_head_fn",
            "hc_head_base",
            "hc_head_scale",
            "norm.",
            "head.",
        ]
    )
    out: Dict[str, torch.Tensor] = {}
    # Pair-dequant within each base prefix that has fp4/fp8 weights.
    for base in ("head.",):
        sub = {k: v for k, v in raw.items() if k.startswith(base)}
        for local_k, v in _dequant_paired(sub, base).items():
            out[base + local_k] = v
    # Plain pass-through for the non-paired ones (norm.weight, hc_head_*).
    for k, v in raw.items():
        if k.startswith("head."):
            continue
        if k.endswith(".scale"):
            continue
        out[k] = v.to(torch.bfloat16) if v.is_floating_point() else v
    return out
