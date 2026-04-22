#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Build a chunked BF16 post-sparse weight cache for GLM-4.7, one layer at a time.

GLM-4.7 weights are already BF16 on HF (no dequant needed). This builder streams
safetensors per-layer, stacks per-expert tensors into the post-sparse layout
(StackedExperts format), and writes one chunk per layer. Peak memory is a few
GB per layer — no swap needed.

Post-sparse layout matches what `enable_sparse_mlp` produces on a freshly built
`Glm4MoeForCausalLM`:
  - Router: `model.layers.N.mlp.gate.*`  -> `model.layers.N.mlp.mlp.router.gate.*`
  - Experts: per-expert  [out,in]  -> stacked+transposed [E, in, out]
             `experts.{0..159}.{gate,up,down}_proj.weight`
             -> `mlp.mlp.experts.{gate,up,down}_proj`
  - Shared experts pass through under `mlp.shared_experts.*`.

Dense layers (i < first_k_dense_replace) pass through unchanged.

Usage:
    python build_weight_cache_glm.py --n-layers 4     # smoke / bringup
    python build_weight_cache_glm.py --n-layers 92    # full GLM-4.7
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

GLM_REPO = "zai-org/GLM-4.7"


def _cache_dir_for(repo_id, n_layers):
    repo_slug = repo_id.replace("/", "--")
    base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(
        base, "tt_xla_dequant_cache", f"{repo_slug}_{n_layers}layers_post_sparse"
    )


def _load_tensors(shard_to_keys, repo_id):
    """Return {ckpt_key: tensor} for all requested keys, opening each shard once."""
    out = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                out[key] = f.get_tensor(key)
    return out


def _group_by_shard(ckpt_keys, weight_map):
    shard_to_keys = {}
    for k in ckpt_keys:
        shard_to_keys.setdefault(weight_map[k], []).append(k)
    return shard_to_keys


def _process_shared(weight_map, repo_id):
    """embed_tokens + norm + lm_head, saved as group `shared`."""
    keys = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
    ]
    keys = [k for k in keys if k in weight_map]
    shard_to_keys = _group_by_shard(keys, weight_map)
    raw = _load_tensors(shard_to_keys, repo_id)
    return {k: raw[k].to(torch.bfloat16) for k in keys}


def _process_dense_layer(layer_idx, weight_map, repo_id):
    """Layer under first_k_dense_replace: pass-through."""
    prefix = f"model.layers.{layer_idx}"
    keys = [k for k in weight_map if k.startswith(prefix + ".")]
    shard_to_keys = _group_by_shard(keys, weight_map)
    raw = _load_tensors(shard_to_keys, repo_id)
    return {k: raw[k].to(torch.bfloat16) for k in keys}


def _process_moe_layer(layer_idx, n_experts, weight_map, repo_id):
    """MoE layer: stack per-expert tensors, rename router/shared/expert prefixes."""
    prefix = f"model.layers.{layer_idx}"
    all_keys = [k for k in weight_map if k.startswith(prefix + ".")]

    expert_keys = []
    other_keys = []
    expert_re = re.compile(
        rf"^model\.layers\.{layer_idx}\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
    )
    for k in all_keys:
        if expert_re.match(k):
            expert_keys.append(k)
        else:
            other_keys.append(k)

    shard_to_keys = _group_by_shard(expert_keys + other_keys, weight_map)
    raw = _load_tensors(shard_to_keys, repo_id)

    # Stack experts. HF stores each as nn.Linear convention [out, in]; TT
    # StackedExperts wants [E, in, out], so we transpose at stack time.
    gate_list, up_list, down_list = (
        [None] * n_experts,
        [None] * n_experts,
        [None] * n_experts,
    )
    for k in expert_keys:
        m = expert_re.match(k)
        idx = int(m.group(1))
        name = m.group(2)
        t = raw[k].to(torch.bfloat16).T.contiguous()  # [in, out]
        if name == "gate_proj":
            gate_list[idx] = t
        elif name == "up_proj":
            up_list[idx] = t
        elif name == "down_proj":
            down_list[idx] = t

    missing_gate = [i for i, x in enumerate(gate_list) if x is None]
    if missing_gate:
        raise ValueError(
            f"Layer {layer_idx}: missing gate_proj for experts {missing_gate[:5]}..."
        )

    gate_proj = torch.stack(gate_list, dim=0)
    up_proj = torch.stack(up_list, dim=0)
    down_proj = torch.stack(down_list, dim=0)
    # Free per-expert list memory before saving.
    del gate_list, up_list, down_list

    state_dict = {
        f"{prefix}.mlp.mlp.experts.gate_proj": gate_proj,
        f"{prefix}.mlp.mlp.experts.up_proj": up_proj,
        f"{prefix}.mlp.mlp.experts.down_proj": down_proj,
    }

    # Non-expert keys: rename router, pass-through shared/attn/norms.
    router_weight_key = f"{prefix}.mlp.gate.weight"
    router_bias_key = f"{prefix}.mlp.gate.e_score_correction_bias"
    for k in other_keys:
        t = raw[k].to(torch.bfloat16)
        if k == router_weight_key:
            new_key = f"{prefix}.mlp.mlp.router.gate.weight"
        elif k == router_bias_key:
            new_key = f"{prefix}.mlp.mlp.router.gate.e_score_correction_bias"
        else:
            new_key = k  # shared_experts, self_attn, norms pass through
        state_dict[new_key] = t

    del raw
    return state_dict


def build_post_sparse_cache(repo_id, n_layers, n_dense_layers, n_experts):
    cache_dir = _cache_dir_for(repo_id, n_layers)

    if os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(cache_dir)
    ):
        print(f"Cache already exists at {cache_dir}, skipping.")
        print(f"  Files: {sorted(os.listdir(cache_dir))}")
        print("  Delete the directory to rebuild.")
        return

    t_total = time.perf_counter()

    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    os.makedirs(cache_dir, exist_ok=True)
    total_bytes = 0

    # Shared group (embed + norm + lm_head)
    t0 = time.perf_counter()
    sd = _process_shared(weight_map, repo_id)
    out_path = os.path.join(cache_dir, "shared.safetensors")
    safetensors_save_file(sd, out_path)
    sz = os.path.getsize(out_path)
    total_bytes += sz
    del sd
    print(f"  shared: {sz / 1e9:.2f} GB, {time.perf_counter() - t0:.1f}s", flush=True)

    # Per-layer groups
    for i in range(n_layers):
        t_layer = time.perf_counter()
        if i < n_dense_layers:
            sd = _process_dense_layer(i, weight_map, repo_id)
            tag = "dense"
        else:
            sd = _process_moe_layer(i, n_experts, weight_map, repo_id)
            tag = "moe"

        out_path = os.path.join(cache_dir, f"layer_{i:04d}.safetensors")
        safetensors_save_file(sd, out_path)
        sz = os.path.getsize(out_path)
        total_bytes += sz
        del sd

        print(
            f"  layer_{i:04d} ({tag}): {sz / 1e9:.2f} GB, "
            f"{time.perf_counter() - t_layer:.1f}s",
            flush=True,
        )

    print(
        f"\nDone. {total_bytes / 1e9:.1f} GB total, "
        f"{time.perf_counter() - t_total:.1f}s",
        flush=True,
    )
    print(f"Cache dir: {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build BF16 post-sparse weight cache for GLM-4.7"
    )
    parser.add_argument("--repo", default=GLM_REPO, help="HuggingFace repo ID")
    parser.add_argument("--n-layers", type=int, default=92, help="Number of layers")
    args = parser.parse_args()

    config_path = hf_hub_download(args.repo, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    n_dense_layers = cfg.get("first_k_dense_replace", 3)
    n_experts = cfg.get("n_routed_experts", 160)
    print(
        f"Repo: {args.repo}\n"
        f"  n_layers={args.n_layers}, n_dense_layers={n_dense_layers}, "
        f"n_experts={n_experts}",
        flush=True,
    )

    build_post_sparse_cache(args.repo, args.n_layers, n_dense_layers, n_experts)
