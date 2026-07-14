#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Convert a quantized DeepSeek-V4 checkpoint to a **bf16, key-preserving**
checkpoint that vLLM loads as an *unquantized* model on TT.

Why: vLLM's DSV4 path uses ``DeepseekV4FP8Config`` (fp8 block-quant linears +
fp4/fp8 MoE experts) which is CUDA-oriented; TT does not do fp8 matmul. Rather
than a fp8-aware quant method, we dequantize the weights to bf16 offline and
drop ``quantization_config`` from ``config.json``. vLLM then builds
``UnquantizedLinearMethod`` / ``UnquantizedFusedMoEMethod`` (→ the plugin's OOT
``TTFusedMoE``) and the bf16 MLA/SWA path — i.e. it lands directly on the
existing TT wiring. This mirrors ``deepseek_v3_2_exp/build_weight_cache.py``,
but **preserves the original tensor names** (that builder renames for the
tt_forge_models torch model; here vLLM's own ``load_weights`` does the mapping)
and **drops the scale tensors** instead of caching them.

The dequant math mirrors ``weight_loader.py`` (which is validated against the
real ``DeepSeek-V4-Flash`` checkpoint): fp8 = e4m3 with a [out/128, in/128]
block scale; fp4 = MXFP4 (e2m1, 2 values/byte) with a [out, in/32] block scale.
Both are a plain block-broadcast multiply.

Usage:
    # From a local snapshot dir (with model.safetensors.index.json):
    python build_vllm_bf16_checkpoint.py --src <snapshot_dir> --dst <bf16_dir>
    # Or download the repo first:
    python build_vllm_bf16_checkpoint.py --repo deepseek-ai/DeepSeek-V4-Flash \
        --dst <bf16_dir>
    # Smoke-convert only the first N layers (won't generate coherent text):
    python build_vllm_bf16_checkpoint.py --src <dir> --dst <dir> --n-layers 2

Then: vllm.LLM(model="<bf16_dir>", ...) with the TT plugin.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, Optional

import torch
from safetensors import safe_open
from safetensors.torch import save_file as _save_file

_FP4_BLOCK = 32
_FP8_BLOCK = 128

# FP4 e2m1fn lookup (4 bits -> float), verbatim from DeepSeek's convert.py.
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

# Scale suffixes seen across DeepSeek checkpoints (native `.scale`, HF fp8
# `.weight_scale_inv`, and the plain `.weight_scale`). All are used as a
# block-broadcast multiplier on the dequantized weight.
_SCALE_SUFFIXES = (".weight_scale_inv", ".weight_scale", ".scale")


# --------------------------------------------------------------------------- #
# Dequant (mirrors weight_loader._dequant_fp8 / _dequant_fp4)
# --------------------------------------------------------------------------- #
def _dequant_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in] fp8_e4m3fn + [out/128, in/128] block scale -> [out, in] bf16."""
    out_dim, in_dim = weight.shape
    assert (
        out_dim % _FP8_BLOCK == 0 and in_dim % _FP8_BLOCK == 0
    ), f"fp8 dims must be multiples of {_FP8_BLOCK}: got {tuple(weight.shape)}"
    w = (
        weight.to(torch.float32)
        .unflatten(0, (-1, _FP8_BLOCK))
        .unflatten(-1, (-1, _FP8_BLOCK))
    )  # [bOut, 128, bIn, 128]
    s = scale.to(torch.float32)[:, None, :, None]  # [bOut, 1, bIn, 1]
    return (w * s).flatten(2, 3).flatten(0, 1).to(torch.bfloat16)


def _dequant_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """[out, in/2] packed fp4 + [out, in/32] block scale -> [out, in] bf16."""
    byte_view = weight.contiguous().view(torch.uint8)
    out_dim, packed_in = byte_view.shape
    in_dim = packed_in * 2
    table = _FP4_TABLE.to(byte_view.device)
    low = (byte_view & 0x0F).long()
    high = ((byte_view >> 4) & 0x0F).long()
    vals = torch.stack([table[low], table[high]], dim=-1).flatten(-2)  # [out, in]
    scale_f = scale.to(torch.float32).repeat_interleave(_FP4_BLOCK, dim=1)
    return (vals * scale_f).to(torch.bfloat16)


def _dequant(weight: torch.Tensor, scale: torch.Tensor, wkey: str) -> torch.Tensor:
    """Dispatch fp4 vs fp8 by the (weight, scale) shape relationship."""
    if scale.ndim == 2 and scale.shape == (
        weight.shape[0],
        weight.shape[1] * 2 // _FP4_BLOCK,
    ):
        return _dequant_fp4(weight, scale)
    if scale.ndim == 2 and scale.shape == (
        weight.shape[0] // _FP8_BLOCK,
        weight.shape[1] // _FP8_BLOCK,
    ):
        return _dequant_fp8(weight, scale)
    raise RuntimeError(
        f"Unrecognized (weight, scale) shapes for {wkey}: "
        f"w={tuple(weight.shape)} s={tuple(scale.shape)}"
    )


# --------------------------------------------------------------------------- #
# Checkpoint I/O
# --------------------------------------------------------------------------- #
def _scale_key_for(wkey: str, present: set) -> Optional[str]:
    """Return the scale key paired with a `.weight` key, if one exists."""
    if not wkey.endswith(".weight"):
        return None
    base = wkey[: -len(".weight")]
    for suf in _SCALE_SUFFIXES:
        if base + suf in present:
            return base + suf
    return None


def _is_scale_key(key: str) -> bool:
    return any(key.endswith(suf) for suf in _SCALE_SUFFIXES)


def _layer_index(key: str) -> Optional[int]:
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _read_index(src: str) -> Dict[str, str]:
    """{tensor_key: shard_filename}. Single-file checkpoints map to one shard."""
    index_path = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]
    # Single-file checkpoint.
    single = "model.safetensors"
    with safe_open(os.path.join(src, single), framework="pt", device="cpu") as f:
        return {k: single for k in f.keys()}


def convert(
    src: str,
    dst: str,
    n_layers: Optional[int] = None,
    present_only: bool = False,
    skip_substrs: Optional[list] = None,
) -> Dict[str, str]:
    """Dequantize `src` -> bf16 `dst`, preserving tensor names, dropping scales,
    and stripping `quantization_config`. Memory-bounded: one shard at a time,
    with cross-shard scale fetch when a scale lives in another shard.

    `present_only`: skip shards whose file is not on disk (for a partially
    downloaded checkpoint, e.g. converting only layers 0-1 without pulling all
    46 shards). Scales always live in the same shard as their weight, so a
    present layer never needs a missing shard.

    `skip_substrs`: drop any tensor whose key contains one of these substrings
    (e.g. ``["mtp."]`` — vLLM's DSV4 loader skips the MTP draft layer, so
    dequantizing it is wasted work).

    Returns the output weight_map (tensor_key -> shard_filename).
    """
    os.makedirs(dst, exist_ok=True)
    skip_substrs = skip_substrs or []
    weight_map = _read_index(src)

    if skip_substrs:
        weight_map = {
            k: shard
            for k, shard in weight_map.items()
            if not any(s in k for s in skip_substrs)
        }
    if present_only:
        weight_map = {
            k: shard
            for k, shard in weight_map.items()
            if os.path.exists(os.path.join(src, shard))
        }
    present = set(weight_map.keys())

    # Group work by source shard so we open each large shard once.
    shard_to_keys: Dict[str, list] = {}
    for k, shard in weight_map.items():
        shard_to_keys.setdefault(shard, []).append(k)

    _scale_cache: Dict[str, torch.Tensor] = {}  # (shard,key) fetched across shards

    def _fetch(shard: str, key: str) -> torch.Tensor:
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    out_weight_map: Dict[str, str] = {}
    for shard in sorted(shard_to_keys):
        keys = shard_to_keys[shard]
        with safe_open(os.path.join(src, shard), framework="pt", device="cpu") as f:
            local = {k: f.get_tensor(k) for k in keys}

        out_tensors: Dict[str, torch.Tensor] = {}
        for k in keys:
            if _is_scale_key(k):
                continue  # scales are consumed by their weight, never emitted
            if n_layers is not None:
                li = _layer_index(k)
                if li is not None and li >= n_layers:
                    continue
            t = local[k]
            skey = _scale_key_for(k, present)
            if skey is not None:
                scale = local.get(skey)
                if scale is None:  # scale lives in another shard
                    scale = _scale_cache.get(skey)
                    if scale is None:
                        scale = _fetch(weight_map[skey], skey)
                        _scale_cache[skey] = scale
                out_tensors[k] = _dequant(t, scale, k)
            else:
                out_tensors[k] = t.to(torch.bfloat16) if t.is_floating_point() else t

        if not out_tensors:
            continue
        out_name = shard if shard.endswith(".safetensors") else "model.safetensors"
        _save_file(out_tensors, os.path.join(dst, out_name))
        for k in out_tensors:
            out_weight_map[k] = out_name

    _write_index(dst, out_weight_map)
    _rewrite_config(src, dst)
    _copy_aux_files(src, dst)
    return out_weight_map


def _write_index(dst: str, weight_map: Dict[str, str]) -> None:
    # Only needed when multi-shard; harmless (and simpler) to always write it.
    shards = set(weight_map.values())
    if len(shards) <= 1:
        # Single-file: vLLM loads model.safetensors directly; no index needed.
        idx = os.path.join(dst, "model.safetensors.index.json")
        if os.path.exists(idx):
            os.remove(idx)
        return
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


def _rewrite_config(src: str, dst: str) -> None:
    """Copy config.json, neutralizing quantization + forcing bf16 dtype.

    ``quantization_config`` is reduced to just ``{"scale_fmt": <orig>}`` rather
    than removed: the DSV4 model reads ``config.quantization_config["scale_fmt"]``
    unconditionally in ``DeepseekV4Attention.__init__``, so the key must exist.
    Dropping ``quant_method`` (and the rest) makes
    ``DeepseekV4FP8Config.override_quantization_method`` return None, so vLLM
    builds the unquantized path (Unquantized{Linear,FusedMoE}Method).
    """
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    orig_qc = cfg.get("quantization_config") or {}
    cfg["quantization_config"] = {"scale_fmt": orig_qc.get("scale_fmt")}
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


def _copy_aux_files(src: str, dst: str) -> None:
    """Copy tokenizer / generation config / etc. — everything that is not a
    safetensors shard, the index, or config.json (handled separately)."""
    skip_exact = {"config.json", "model.safetensors.index.json"}
    for name in os.listdir(src):
        if name in skip_exact or name.endswith(".safetensors"):
            continue
        s = os.path.join(src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(dst, name))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="Local snapshot dir (with the safetensors).")
    ap.add_argument("--repo", help="HF repo to download first (if --src absent).")
    ap.add_argument("--dst", required=True, help="Output bf16 checkpoint dir.")
    ap.add_argument(
        "--n-layers", type=int, default=None, help="Convert only the first N layers."
    )
    ap.add_argument(
        "--present-only",
        action="store_true",
        help="Skip shards not on disk (for a partially downloaded checkpoint).",
    )
    ap.add_argument(
        "--skip-substr",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="Drop tensors whose key contains SUBSTR (repeatable), e.g. 'mtp.'.",
    )
    args = ap.parse_args()

    src = args.src
    if src is None:
        if not args.repo:
            ap.error("one of --src or --repo is required")
        from huggingface_hub import snapshot_download

        src = snapshot_download(args.repo)
    print(f"Converting {src} -> {args.dst} (bf16, unquantized)")
    wm = convert(
        src,
        args.dst,
        n_layers=args.n_layers,
        present_only=args.present_only,
        skip_substrs=args.skip_substr,
    )
    print(f"Done: {len(wm)} tensors, {len(set(wm.values()))} shard(s) in {args.dst}")


if __name__ == "__main__":
    main()
