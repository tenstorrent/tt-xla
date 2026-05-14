# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
from typing import Dict, Iterable, List, Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from third_party.tt_forge_models.deepseek_v4.modified_model import model

# FP4 e2m1fn lookup: 4 bits -> float value. Bits 0-7 positive, 8-15 negative
# (bit 3 acts as sign). Copied verbatim from https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/inference/convert.py 
_FP4_TABLE = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

_FP4_BLOCK = 32
_FP8_BLOCK = 128

_VALID_MODEL_NAMES = ('deepseek-ai/DeepSeek-V4-Flash', 'deepseek-ai/DeepSeek-V4-Pro')

##################################################################
# Dequanting helper functions START
##################################################################
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
##################################################################
# Dequanting helper functions END
##################################################################


##################################################################
# Weight loading helper functions START 
##################################################################
def _find_shards_for_keys(
    weight_map: Dict[str, str], prefixes: Iterable[str]
) -> List[str]:
    prefixes = tuple(prefixes)
    return sorted({shard for k, shard in weight_map.items() if k.startswith(prefixes)})


def _load_raw_subset(
    model_name: str, prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    """Download relevant shards and return tensors whose keys match any prefix.
    Keys are returned verbatim (with the original "layers.{L}.ffn." prefix)."""
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
##################################################################
# Weight loading helper functions END 
##################################################################

def load_config_args(model_name: str, force_bf16: bool) -> model.ModelArgs:
    path = hf_hub_download(repo_id=model_name, filename="inference/config.json")
    with open(path) as f:
        raw = json.load(f)

    valid = {f.name for f in model.ModelArgs.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in raw.items() if k in valid}
    # compress_ratios arrives as a list; the dataclass expects a tuple.
    if "compress_ratios" in kwargs:
        kwargs["compress_ratios"] = tuple(kwargs["compress_ratios"])
    if force_bf16:
        # Force bf16 execution path: no FP4 experts, no FP8 non-expert linears.
        kwargs["dtype"] = "bf16"
        kwargs["expert_dtype"] = None
        kwargs["scale_fmt"] = None
    return model.ModelArgs(**kwargs)

def load_moe_state_dict(model_name: str, layer_id: int) -> Dict[str, torch.Tensor]:
    """State dict matching `model.MoE(layer_id, args).state_dict()` keys.

    Returns keys like: gate.weight, gate.bias, experts.{E}.w{1,2,3}.weight,
    shared_experts.w{1,2,3}.weight — all in bf16 (except gate.bias which
    remains in its checkpoint dtype, typically fp32).
    """
    base = f"layers.{layer_id}.ffn."
    raw = _load_raw_subset(model_name, [base])
    return _dequant_paired(raw, base)

def load_expert_state_dict(model_name: str, layer_id: int, expert_id: int) -> Dict[str, torch.Tensor]:
    """State dict matching `model.Expert(...).state_dict()` keys.

    Returns keys: w1.weight, w2.weight, w3.weight — bf16.
    """
    base = f"layers.{layer_id}.ffn.experts.{expert_id}."
    raw = _load_raw_subset(model_name, [base])
    return _dequant_paired(raw, base)


def load_block_state_dict(model_name: str, layer_id: int) -> Dict[str, torch.Tensor]:
    """State dict matching `model.Block(layer_id, args).state_dict()` keys.

    Pulls the full Block: attention (incl. compressor/indexer), attn_norm,
    ffn (MoE — gate, experts, shared), ffn_norm, hc_attn_*, hc_ffn_*.

    Heavy: hash-routed layers contain 256 routed experts × 3 weight matrices
    (fp4-packed in storage; ~3GB on disk per layer, ~12GB after bf16 dequant
    in memory).
    """
    base = f"layers.{layer_id}."
    raw = _load_raw_subset(model_name, [base])
    return _dequant_paired(raw, base)


def load_embed_state_dict(model_name: str) -> Dict[str, torch.Tensor]:
    """Top-level embedding weights for `Transformer.embed`.

    Returns: {"weight": tensor[vocab_size, dim] bf16}.
    """
    raw = _load_raw_subset(model_name, ["embed."])
    return _dequant_paired(raw, "embed.")


def load_top_level_state_dict(model_name: str) -> Dict[str, torch.Tensor]:
    """Top-level Transformer params that don't live under a layer.

    Includes hc_head_fn / hc_head_base / hc_head_scale (the head-side
    Hyper-Connections params that reduce hc_mult copies back to one), the
    final RMSNorm `norm.weight`, and the parallel head's `head.weight`.
    Keys are returned verbatim with their top-level dotted names.
    """
    raw = _load_raw_subset(
        model_name,
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

def load_transformer_state_dict(
    model_name: str,
    layer_ids: Iterable[int],
    include_mtp: bool = False,
) -> Dict[str, torch.Tensor]:
    """Full Transformer state dict for the requested layer subset plus
    top-level (embed, norm, head, hc_head_*). Load with strict=False —
    non-persistent buffers (kv_cache, freqs_cis) aren't in the checkpoint.
    """
    layer_ids = sorted(set(layer_ids))
    prefixes: List[str] = ["embed.", "norm.", "head.", "hc_head_"]
    prefixes.extend(f"layers.{L}." for L in layer_ids)
    if include_mtp:
        prefixes.append("mtp.")
    raw = _load_raw_subset(model_name, prefixes)
    return _dequant_paired(raw, "")



