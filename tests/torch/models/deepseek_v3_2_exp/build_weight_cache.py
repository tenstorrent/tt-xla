#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Build a chunked BF16 weight cache for DeepSeek V3.1, one layer at a time.

Processes each layer independently so peak memory is ~23 GB (one MoE layer)
instead of ~1.37 TB (all 61 layers). No swap needed on a 512 GB machine.

Usage:
    python build_weight_cache.py                  # default: 61 layers
    python build_weight_cache.py --n-layers 10
    python build_weight_cache.py --n-layers 4 --repo deepseek-ai/DeepSeek-V3.1
"""
import argparse
import json
import os
import re
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file as safetensors_save_file

# Import the canonical helpers from the test file to stay in sync
sys.path.insert(0, os.path.dirname(__file__))
from test_deepseek_v3_2_exp import _rename_hf_key, _weight_dequant

DEEPSEEK_V3_1_REPO = "deepseek-ai/DeepSeek-V3.1"


def build_cache(repo_id, n_layers, n_dense_layers=1):
    repo_slug = repo_id.replace("/", "--")
    base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.path.join(
        base, "tt_xla_dequant_cache", f"{repo_slug}_{n_layers}layers"
    )

    if os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(cache_dir)
    ):
        print(f"Cache already exists at {cache_dir}, skipping.")
        print(f"  Files: {sorted(os.listdir(cache_dir))}")
        print("  Delete the directory to rebuild.")
        return

    t_total_start = time.perf_counter()

    # Load index
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Build per-layer mapping: which ckpt keys (and their scales) belong to
    # which layer/group.
    # Groups: "shared" (embed, norm, head) and layer_0000..layer_NNNN
    groups = {}  # group_name -> {ckpt_key: model_key}
    scale_keys = {}  # ckpt_weight_key -> ckpt_scale_key
    all_key_to_shard = {}  # ckpt_key -> shard_file (weights + scales)

    for ckpt_key, shard_file in weight_map.items():
        layer_m = re.match(r"model\.layers\.(\d+)\.", ckpt_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue

        if "weight_scale_inv" in ckpt_key:
            w_key = ckpt_key.replace(".weight_scale_inv", ".weight")
            scale_keys[w_key] = ckpt_key
            all_key_to_shard[ckpt_key] = shard_file
        else:
            model_key = _rename_hf_key(ckpt_key, n_dense_layers)
            if model_key is None:
                continue
            all_key_to_shard[ckpt_key] = shard_file

            # Assign to group
            lm = re.match(r"layers\.(\d+)\.", model_key)
            group = f"layer_{int(lm.group(1)):04d}" if lm else "shared"
            groups.setdefault(group, {})[ckpt_key] = model_key

    os.makedirs(cache_dir, exist_ok=True)
    total_bytes = 0

    for group_name in sorted(groups):
        t_group_start = time.perf_counter()
        ckpt_to_model = groups[group_name]

        # Determine which shards we need for this group
        needed_shard_files = set()
        for ckpt_key in ckpt_to_model:
            needed_shard_files.add(all_key_to_shard[ckpt_key])
            if ckpt_key in scale_keys:
                needed_shard_files.add(all_key_to_shard[scale_keys[ckpt_key]])

        # Load only the tensors we need from those shards
        raw = {}
        needed_ckpt_keys = set(ckpt_to_model.keys())
        needed_scale_keys = {scale_keys[k] for k in ckpt_to_model if k in scale_keys}
        all_needed = needed_ckpt_keys | needed_scale_keys

        for shard_name in sorted(needed_shard_files):
            shard_path = hf_hub_download(repo_id, shard_name)
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in all_needed:
                        raw[key] = f.get_tensor(key)

        # Dequantize and rename
        state_dict = {}
        n_dequant = 0
        for ckpt_key, model_key in ckpt_to_model.items():
            tensor = raw.get(ckpt_key)
            if tensor is None:
                continue

            if tensor.dtype == torch.float8_e4m3fn:
                sk = scale_keys.get(ckpt_key)
                scale_inv = raw.get(sk) if sk else None
                if scale_inv is not None:
                    tensor = _weight_dequant(tensor, scale_inv)
                    n_dequant += 1
                else:
                    tensor = tensor.to(torch.bfloat16)

            if model_key == "head.weight":
                tensor = tensor.to(torch.float32)
            elif tensor.dtype != torch.bfloat16:
                tensor = tensor.to(torch.bfloat16)

            state_dict[model_key] = tensor

        del raw

        # Save chunk
        chunk_path = os.path.join(cache_dir, f"{group_name}.safetensors")
        safetensors_save_file(state_dict, chunk_path)
        chunk_size = os.path.getsize(chunk_path)
        total_bytes += chunk_size
        del state_dict

        t_group_end = time.perf_counter()
        print(
            f"  {group_name}: {len(ckpt_to_model)} keys, "
            f"{n_dequant} dequantized, "
            f"{chunk_size / 1e9:.1f} GB, "
            f"{t_group_end - t_group_start:.1f}s",
            flush=True,
        )

    t_total_end = time.perf_counter()
    print(
        f"\nDone. {len(groups)} chunks, {total_bytes / 1e9:.1f} GB total, "
        f"{t_total_end - t_total_start:.1f}s",
        flush=True,
    )
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build BF16 weight cache for DeepSeek V3.1"
    )
    parser.add_argument(
        "--repo", default=DEEPSEEK_V3_1_REPO, help="HuggingFace repo ID"
    )
    parser.add_argument("--n-layers", type=int, default=61, help="Number of layers")
    parser.add_argument(
        "--n-dense-layers",
        type=int,
        default=None,
        help="Number of dense (non-MoE) layers (default: read from config)",
    )
    args = parser.parse_args()

    n_dense_layers = args.n_dense_layers
    if n_dense_layers is None:
        config_path = hf_hub_download(args.repo, "config.json")
        with open(config_path) as f:
            hf_cfg = json.load(f)
        n_dense_layers = hf_cfg.get("first_k_dense_replace", 1)
        print(f"Read n_dense_layers={n_dense_layers} from {args.repo} config")

    build_cache(args.repo, args.n_layers, n_dense_layers)
