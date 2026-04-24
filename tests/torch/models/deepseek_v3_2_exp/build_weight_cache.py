#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
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
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

DEEPSEEK_V3_1_REPO = "deepseek-ai/DeepSeek-V3.1"
FP8_BLOCK_SIZE = 128


def _dequant_cache_dir(repo_id, n_layers):
    """Per-model BF16 safetensors cache directory."""
    repo_slug = repo_id.replace("/", "--")
    base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(base, "tt_xla_dequant_cache", f"{repo_slug}_{n_layers}layers")


def _post_sparse_cache_dir(repo_id, n_layers):
    return _dequant_cache_dir(repo_id, n_layers) + "_post_sparse"


def _has_cache(cache_dir):
    return os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(cache_dir)
    )


def _rename_hf_key(ckpt_key, n_dense_layers=1):
    """Rename a HuggingFace checkpoint key to match modified_model.py naming."""
    key = ckpt_key
    if key.startswith("model."):
        key = key[len("model.") :]
    if "weight_scale_inv" in key:
        return None
    key = key.replace("lm_head.", "head.")
    key = key.replace("embed_tokens.", "embed.")
    key = re.sub(r"(layers\.\d+\.)input_layernorm\.", r"\1attn_norm.", key)
    key = re.sub(r"(layers\.\d+\.)post_attention_layernorm\.", r"\1ffn_norm.", key)
    key = key.replace("self_attn.indexer.", "attn.indexer.")
    key = key.replace("self_attn.q_a_proj.", "attn.wq_a.")
    key = key.replace("self_attn.q_b_proj.", "attn.wq_b.")
    key = key.replace("self_attn.q_a_layernorm.", "attn.q_norm.")
    key = key.replace("self_attn.kv_a_proj_with_mqa.", "attn.wkv_a.")
    key = key.replace("self_attn.kv_b_proj.", "attn.wkv_b.")
    key = key.replace("self_attn.kv_a_layernorm.", "attn.kv_norm.")
    key = key.replace("self_attn.o_proj.", "attn.wo.")
    key = re.sub(r"mlp\.experts\.(\d+)\.gate_proj\.", r"ffn.experts.\1.w1.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.down_proj\.", r"ffn.experts.\1.w2.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.up_proj\.", r"ffn.experts.\1.w3.", key)
    key = key.replace("mlp.shared_experts.gate_proj.", "ffn.shared_experts.w1.")
    key = key.replace("mlp.shared_experts.down_proj.", "ffn.shared_experts.w2.")
    key = key.replace("mlp.shared_experts.up_proj.", "ffn.shared_experts.w3.")
    key = key.replace("mlp.gate.e_score_correction_bias", "mlp.gate.bias")
    key = key.replace("mlp.gate.", "ffn.gate.")
    layer_m = re.match(r"layers\.(\d+)\.", key)
    if layer_m:
        layer_id = int(layer_m.group(1))
        if layer_id < n_dense_layers:
            key = key.replace("mlp.gate_proj.", "ffn.w1.")
            key = key.replace("mlp.down_proj.", "ffn.w2.")
            key = key.replace("mlp.up_proj.", "ffn.w3.")
        elif (
            "mlp.gate_proj." in key or "mlp.down_proj." in key or "mlp.up_proj." in key
        ):
            return None
    return key


def _weight_dequant(weight, scale_inv, block_size=FP8_BLOCK_SIZE):
    """Dequantize FP8 weight: bf16 = fp8 * weight_scale_inv (block-wise)."""
    orig_shape = weight.shape
    assert weight.dim() == 2
    rows, cols = orig_shape
    pad_rows = (block_size - rows % block_size) % block_size
    pad_cols = (block_size - cols % block_size) % block_size
    if pad_rows or pad_cols:
        weight = torch.nn.functional.pad(weight, (0, pad_cols, 0, pad_rows))
    padded_rows, padded_cols = weight.shape
    n_br = padded_rows // block_size
    n_bc = padded_cols // block_size
    weight = (
        weight.view(n_br, block_size, n_bc, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(-1, block_size * block_size)
    )
    weight = (
        (weight.float() * scale_inv.view(-1, 1).float())
        .to(torch.bfloat16)
        .view(n_br, n_bc, block_size, block_size)
        .transpose(1, 2)
        .contiguous()
        .view(padded_rows, padded_cols)
    )
    return weight[:rows, :cols]


def _group_by_shard(ckpt_keys, key_to_shard):
    shard_to_keys = {}
    for k in ckpt_keys:
        shard_to_keys.setdefault(key_to_shard[k], []).append(k)
    return shard_to_keys


def _load_tensors(shard_to_keys, repo_id):
    """Return {ckpt_key: tensor} for all requested keys, opening each shard once."""
    out = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in keys:
                out[key] = f.get_tensor(key)
    return out


def build_cache(repo_id, n_layers, n_dense_layers=1):
    cache_dir = _dequant_cache_dir(repo_id, n_layers)

    if _has_cache(cache_dir):
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

        needed_ckpt_keys = set(ckpt_to_model.keys())
        needed_scale_keys = {scale_keys[k] for k in ckpt_to_model if k in scale_keys}
        shard_to_keys = _group_by_shard(
            needed_ckpt_keys | needed_scale_keys, all_key_to_shard
        )
        raw = _load_tensors(shard_to_keys, repo_id)

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


def _stack_experts_for_chunk(chunk):
    """Convert per-expert weights in a chunk to stacked StackedExperts format.

    Input keys like ``layers.N.ffn.experts.{idx}.{w1,w2,w3}.weight``
    become ``layers.N.ffn.mlp.experts.{gate,up,down}_proj`` (transposed & stacked).
    Router keys ``ffn.gate.*`` become ``ffn.mlp.router.gate.*``.
    All other keys (attention, norms, shared_experts) pass through unchanged.
    """
    # Collect per-expert weights
    expert_weights = {}  # idx -> {w1: tensor, w2: tensor, w3: tensor}
    layer_prefix = None

    for key in chunk:
        m = re.match(r"(layers\.\d+\.ffn)\.experts\.(\d+)\.(w[123])\.weight", key)
        if m:
            layer_prefix = m.group(1)
            idx, wname = int(m.group(2)), m.group(3)
            expert_weights.setdefault(idx, {})[wname] = chunk[key]

    if not expert_weights:
        return dict(chunk)

    n_experts = max(expert_weights.keys()) + 1
    result = {}

    # Stack: w1=gate, w3=up, w2=down — with transpose to match StackedExperts
    gate_proj = torch.stack([expert_weights[i]["w1"].T for i in range(n_experts)])
    up_proj = torch.stack([expert_weights[i]["w3"].T for i in range(n_experts)])
    down_proj = torch.stack([expert_weights[i]["w2"].T for i in range(n_experts)])

    inter = gate_proj.shape[-1]
    hidden = gate_proj.shape[1]
    dtype = gate_proj.dtype

    mlp_pfx = f"{layer_prefix}.mlp.experts"
    result[f"{mlp_pfx}.gate_proj"] = gate_proj
    result[f"{mlp_pfx}.up_proj"] = up_proj
    result[f"{mlp_pfx}.down_proj"] = down_proj
    result[f"{mlp_pfx}.gate_proj_bias"] = torch.zeros(n_experts, inter, dtype=dtype)
    result[f"{mlp_pfx}.up_proj_bias"] = torch.zeros(n_experts, inter, dtype=dtype)
    result[f"{mlp_pfx}.down_proj_bias"] = torch.zeros(n_experts, hidden, dtype=dtype)

    # Non-expert keys: rename router, pass through rest
    for key in chunk:
        if ".ffn.experts." in key:
            continue  # already handled
        if ".ffn.gate." in key:
            new_key = key.replace(".ffn.gate.", ".ffn.mlp.router.gate.")
            result[new_key] = chunk[key]
        else:
            result[key] = chunk[key]

    return result


def build_post_sparse_cache(repo_id, n_layers, n_dense_layers):
    """Build post-sparse cache from existing pre-sparse cache.

    Reads each layer's pre-sparse chunk via mmap, stacks expert weights
    to match the StackedExperts layout, and saves. Peak memory is ~46 GB
    (one layer of mmap'd per-expert weights + one layer of stacked tensors).
    """
    pre_dir = _dequant_cache_dir(repo_id, n_layers)
    post_dir = _post_sparse_cache_dir(repo_id, n_layers)

    if not _has_cache(pre_dir):
        print(
            f"Pre-sparse cache not found at {pre_dir}, building it first...",
            flush=True,
        )
        build_cache(repo_id, n_layers, n_dense_layers)

    if _has_cache(post_dir):
        print(f"Post-sparse cache already exists at {post_dir}, skipping.")
        print(f"  Files: {sorted(os.listdir(post_dir))}")
        print("  Delete the directory to rebuild.")
        return

    os.makedirs(post_dir, exist_ok=True)
    total_bytes = 0
    t_start = time.perf_counter()

    for fname in sorted(os.listdir(pre_dir)):
        if not fname.endswith(".safetensors"):
            continue

        t_chunk = time.perf_counter()
        chunk = safetensors_load_file(os.path.join(pre_dir, fname))

        is_moe = any(".ffn.experts." in k for k in chunk)
        if is_moe:
            out_dict = _stack_experts_for_chunk(chunk)
        else:
            out_dict = dict(chunk)

        out_path = os.path.join(post_dir, fname)
        safetensors_save_file(out_dict, out_path)
        sz = os.path.getsize(out_path)
        total_bytes += sz
        del chunk, out_dict

        dt = time.perf_counter() - t_chunk
        tag = " (MoE stacked)" if is_moe else ""
        print(f"  {fname}: {sz / 1e9:.1f} GB, {dt:.1f}s{tag}", flush=True)

    dt_total = time.perf_counter() - t_start
    print(
        f"\nDone. {total_bytes / 1e9:.1f} GB total, {dt_total:.1f}s",
        flush=True,
    )
    print(f"Post-sparse cache dir: {post_dir}")


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
    parser.add_argument(
        "--post-sparse",
        action="store_true",
        help="Build post-sparse cache (stacked experts). Requires pre-sparse cache.",
    )
    args = parser.parse_args()

    n_dense_layers = args.n_dense_layers
    if n_dense_layers is None:
        config_path = hf_hub_download(args.repo, "config.json")
        with open(config_path) as f:
            hf_cfg = json.load(f)
        n_dense_layers = hf_cfg["first_k_dense_replace"]
        print(f"Read n_dense_layers={n_dense_layers} from {args.repo} config")

    if args.post_sparse:
        build_post_sparse_cache(args.repo, args.n_layers, n_dense_layers)
    else:
        build_cache(args.repo, args.n_layers, n_dense_layers)
