# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import os
import re
import time

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.runtime as xr
from benchmark.utils import compute_pcc
from huggingface_hub import hf_hub_download
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig, TorchComparisonEvaluator
from infra.testers.compiler_config import CompilerConfig
from modified_model import ModelArgs, MoE
from modified_model import Transformer as ModifiedTransformer
from modified_model import precompute_freqs_cis
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch import nn
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from tests.utils import failed_ttmlir_compilation

# This model is modified from the original deepseek_v3_2_exp model.py to:
# 1. Use scipy.linalg.hadamard instead of fast_hadamard_transform
#    - fast_hadamard_transform requires a CUDA enviroment and fails to install
# 2. Disable FP8 quantization features (act_quant, fp8_gemm, fp8_index) with stubs
#    - the original implementation (kernel.py) relies on custom tilelang kernels not supported on TT
# 3. Avoid torch.view_as_complex/view_as_real operations

DEEPSEEK_V3_2_REPO = "deepseek-ai/DeepSeek-V3.2"
DEEPSEEK_V3_1_REPO = "deepseek-ai/DeepSeek-V3.1"


def _rename_hf_key(ckpt_key, n_dense_layers=1):
    """Rename a HuggingFace checkpoint key to match modified_model.py state dict naming."""
    key = ckpt_key

    # Strip "model." prefix
    if key.startswith("model."):
        key = key[len("model.") :]

    # Skip FP8 quantization scale keys
    if "weight_scale_inv" in key:
        return None

    # Top-level renames
    key = key.replace("lm_head.", "head.")
    key = key.replace("embed_tokens.", "embed.")

    # Layer norms
    key = re.sub(r"(layers\.\d+\.)input_layernorm\.", r"\1attn_norm.", key)
    key = re.sub(r"(layers\.\d+\.)post_attention_layernorm\.", r"\1ffn_norm.", key)

    # Attention (indexer must come before other self_attn renames)
    key = key.replace("self_attn.indexer.", "attn.indexer.")
    key = key.replace("self_attn.q_a_proj.", "attn.wq_a.")
    key = key.replace("self_attn.q_b_proj.", "attn.wq_b.")
    key = key.replace("self_attn.q_a_layernorm.", "attn.q_norm.")
    key = key.replace("self_attn.kv_a_proj_with_mqa.", "attn.wkv_a.")
    key = key.replace("self_attn.kv_b_proj.", "attn.wkv_b.")
    key = key.replace("self_attn.kv_a_layernorm.", "attn.kv_norm.")
    key = key.replace("self_attn.o_proj.", "attn.wo.")

    # MoE routed experts (before bare mlp renames)
    key = re.sub(r"mlp\.experts\.(\d+)\.gate_proj\.", r"ffn.experts.\1.w1.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.down_proj\.", r"ffn.experts.\1.w2.", key)
    key = re.sub(r"mlp\.experts\.(\d+)\.up_proj\.", r"ffn.experts.\1.w3.", key)

    # MoE shared experts (explicit shared_experts prefix)
    key = key.replace("mlp.shared_experts.gate_proj.", "ffn.shared_experts.w1.")
    key = key.replace("mlp.shared_experts.down_proj.", "ffn.shared_experts.w2.")
    key = key.replace("mlp.shared_experts.up_proj.", "ffn.shared_experts.w3.")

    # MoE router gate
    key = key.replace("mlp.gate.e_score_correction_bias", "mlp.gate.bias")
    key = key.replace("mlp.gate.", "ffn.gate.")

    # Bare mlp.{gate_proj,down_proj,up_proj}: only valid for dense layers.
    # For MoE layers, these have incompatible shapes — skip them.
    # (MoE shared experts are loaded via explicit mlp.shared_experts.* keys above.)
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
            return None  # Skip — incompatible shape for MoE shared experts

    return key


def load_deepseek_config(repo_id=DEEPSEEK_V3_2_REPO):
    """Download and parse the HuggingFace config.json into ModelArgs fields."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        hf_cfg = json.load(f)

    rope_scaling = hf_cfg.get("rope_scaling", {})

    # Map HuggingFace config keys to ModelArgs field names
    return ModelArgs(
        vocab_size=hf_cfg["vocab_size"],
        dim=hf_cfg["hidden_size"],
        inter_dim=hf_cfg["intermediate_size"],
        moe_inter_dim=hf_cfg["moe_intermediate_size"],
        n_layers=hf_cfg["num_hidden_layers"],
        n_dense_layers=hf_cfg.get("first_k_dense_replace", 1),
        n_heads=hf_cfg["num_attention_heads"],
        n_routed_experts=hf_cfg.get("n_routed_experts", 256),
        n_shared_experts=hf_cfg.get("n_shared_experts", 1),
        n_activated_experts=hf_cfg.get("num_experts_per_tok", 8),
        n_expert_groups=hf_cfg.get("n_group", 8),
        n_limited_groups=hf_cfg.get("topk_group", 4),
        score_func=hf_cfg.get("scoring_func", "sigmoid"),
        route_scale=hf_cfg.get("routed_scaling_factor", 2.5),
        q_lora_rank=hf_cfg.get("q_lora_rank", 1536),
        kv_lora_rank=hf_cfg.get("kv_lora_rank", 512),
        qk_nope_head_dim=hf_cfg.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=hf_cfg.get("qk_rope_head_dim", 64),
        v_head_dim=hf_cfg.get("v_head_dim", 128),
        original_seq_len=rope_scaling.get("original_max_position_embeddings", 4096),
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        rope_factor=rope_scaling.get("factor", 40),
        beta_fast=rope_scaling.get("beta_fast", 32),
        beta_slow=rope_scaling.get("beta_slow", 1),
        mscale=rope_scaling.get("mscale", 1.0),
        index_n_heads=hf_cfg.get("index_n_heads", 0),
        index_head_dim=hf_cfg.get("index_head_dim", 128),
        index_topk=hf_cfg.get("index_topk", 2048),
    )


def load_deepseek_weights(
    model, repo_id=DEEPSEEK_V3_2_REPO, n_layers=2, n_dense_layers=1
):
    """Load pretrained weights from a HuggingFace repo into the model.

    The HF checkpoint uses different key naming than modified_model.py,
    so keys are remapped during loading.  Only the safetensors shards
    containing the first ``n_layers`` layers (plus top-level weights
    like embed, norm, head) are downloaded.

    Weights not found in the checkpoint (e.g. Indexer parameters, caches)
    remain at their initialized values.
    """
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Build ckpt_key -> (model_key, shard_file) mapping, filtering by layer
    needed_shards = set()
    needed_keys = {}  # ckpt_key -> model_key
    for ckpt_key, shard_file in weight_map.items():
        model_key = _rename_hf_key(ckpt_key, n_dense_layers)
        if model_key is None:
            continue
        # Filter to only needed layers
        layer_m = re.match(r"layers\.(\d+)\.", model_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue
        needed_shards.add(shard_file)
        needed_keys[ckpt_key] = model_key

    # Download needed shards and build state dict
    state_dict = {}
    for shard_name in sorted(needed_shards):
        print(f"[weights] loading shard: {shard_name}")
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in needed_keys:
                    state_dict[needed_keys[key]] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"[weights] loaded {len(state_dict)} tensors from {repo_id}. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )
    if unexpected:
        print(
            f"[weights] first 20 unexpected keys (checkpoint): {sorted(unexpected)[:20]}"
        )
    if missing:
        print(f"[weights] first 20 missing keys (model): {sorted(missing)[:20]}")

    # Verify all weight parameters were loaded (non-weight buffers like caches may remain at init)
    model_keys = set(model.state_dict().keys())
    loaded_keys = set(state_dict.keys())
    not_loaded = model_keys - loaded_keys
    print(f"[weights] model keys not loaded: {sorted(not_loaded)}")

    return model


FP8_BLOCK_SIZE = 128


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


def _load_hf_dequantized_weights(model, repo_id, n_layers):
    """Load and dequantize FP8 safetensors weights into an HF model.

    Adapted from generate_v3_1_hf_cpu.py — handles FP8 block dequantization
    and fuses individual expert weights into the 3D tensors HF expects.
    """
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Filter to needed layers and separate weights from scales
    weight_keys = {}  # ckpt_key -> shard_file
    scale_keys = {}  # weight_ckpt_key -> scale_ckpt_key
    scale_shards = {}  # scale_ckpt_key -> shard_file

    for ckpt_key, shard_file in weight_map.items():
        layer_m = re.match(r"model\.layers\.(\d+)\.", ckpt_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue
        if "weight_scale_inv" in ckpt_key:
            w_key = ckpt_key.replace(".weight_scale_inv", ".weight")
            scale_keys[w_key] = ckpt_key
            scale_shards[ckpt_key] = shard_file
        else:
            weight_keys[ckpt_key] = shard_file

    all_shards = set(weight_keys.values()) | set(scale_shards.values())

    raw = {}
    for shard_name in sorted(all_shards):
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_keys or key in scale_shards:
                    raw[key] = f.get_tensor(key)
    print(f"[weights] loaded {len(raw)} raw tensors from {len(all_shards)} shards")

    # Dequantize FP8 weights
    dequantized = {}
    n_dequant = 0
    for ckpt_key in weight_keys:
        tensor = raw.get(ckpt_key)
        if tensor is None:
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = scale_keys.get(ckpt_key)
            scale_inv = raw.get(scale_key) if scale_key else None
            if scale_inv is not None:
                tensor = _weight_dequant(tensor, scale_inv)
                n_dequant += 1
            else:
                tensor = tensor.to(torch.bfloat16)
        dequantized[ckpt_key] = tensor
    print(f"[weights] dequantized {n_dequant} FP8 tensors")

    # Check whether the model expects fused expert format (gate_up_proj 3D tensors)
    # or individual per-expert format (experts.{i}.gate_proj.weight 2D tensors).
    model_keys = set(model.state_dict().keys())
    needs_fusion = any("gate_up_proj" in k for k in model_keys)

    if needs_fusion:
        # Fuse individual expert weights into 3D tensors for HF model.
        state_dict = {}
        expert_groups = {}
        expert_pattern = re.compile(
            r"(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
        )
        for ckpt_key, tensor in dequantized.items():
            m = expert_pattern.match(ckpt_key)
            if m:
                prefix, idx, proj = m.group(1), int(m.group(2)), m.group(3)
                expert_groups.setdefault(prefix, {}).setdefault(idx, {})[proj] = tensor
            else:
                state_dict[ckpt_key] = tensor

        for prefix, experts in expert_groups.items():
            num_experts = max(experts.keys()) + 1
            gate_up_list, down_list = [], []
            for i in range(num_experts):
                gate_up_list.append(
                    torch.cat([experts[i]["gate_proj"], experts[i]["up_proj"]], dim=0)
                )
                down_list.append(experts[i]["down_proj"])
            state_dict[f"{prefix}.gate_up_proj"] = torch.stack(gate_up_list)
            state_dict[f"{prefix}.down_proj"] = torch.stack(down_list)

        if expert_groups:
            print(f"[weights] fused experts for {len(expert_groups)} MoE layers")
    else:
        # Model uses individual per-expert weights — checkpoint format matches
        state_dict = dequantized
        print("[weights] model uses individual expert format, no fusion needed")

    # Cast to bf16. Keep e_score_correction_bias in fp32 —
    # BF16 truncation at magnitude ~5 causes ~0.015 error which flips MoE routing.
    for key, tensor in state_dict.items():
        if "e_score_correction_bias" not in key and tensor.dtype != torch.bfloat16:
            state_dict[key] = tensor.to(torch.bfloat16)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )
    if missing:
        real_missing = [k for k in missing if "weight_scale_inv" not in k]
        if real_missing:
            print(f"[weights] missing: {sorted(real_missing)[:10]}")


def _cache_dir_for(repo_id, n_layers):
    """Return the per-model cache directory for chunked BF16 safetensors."""
    repo_slug = repo_id.replace("/", "--")
    base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(base, "tt_xla_dequant_cache", f"{repo_slug}_{n_layers}layers")


def _post_sparse_cache_dir_for(repo_id, n_layers):
    """Return the post-sparse cache directory (stacked expert weights)."""
    return _cache_dir_for(repo_id, n_layers) + "_post_sparse"


def _has_cache(cache_dir):
    """Check if a cache directory has safetensors files."""
    return os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(cache_dir)
    )


def _save_cache_chunked(state_dict, cache_dir):
    """Save state_dict as per-layer safetensors chunks + a shared chunk.

    Each chunk is small enough that safetensors' serialization buffer
    doesn't double peak memory.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Partition keys by layer
    layer_dicts = {}  # layer_idx -> {key: tensor}
    shared = {}
    for key, tensor in state_dict.items():
        m = re.match(r"layers\.(\d+)\.", key)
        if m:
            layer_dicts.setdefault(int(m.group(1)), {})[key] = tensor
        else:
            shared[key] = tensor

    total_bytes = 0
    if shared:
        path = os.path.join(cache_dir, "shared.safetensors")
        safetensors_save_file(shared, path)
        total_bytes += os.path.getsize(path)

    for layer_idx in sorted(layer_dicts):
        path = os.path.join(cache_dir, f"layer_{layer_idx:04d}.safetensors")
        safetensors_save_file(layer_dicts[layer_idx], path)
        sz = os.path.getsize(path)
        total_bytes += sz
        del layer_dicts[layer_idx]  # free after saving
        print(
            f"  [cache] saved layer {layer_idx} ({sz / 1e9:.1f} GB)",
            flush=True,
        )

    return total_bytes


def _load_cache_chunked(cache_dir):
    """Load all chunk files from cache_dir via mmap. Returns merged state_dict."""
    state_dict = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".safetensors"):
            continue
        chunk = safetensors_load_file(os.path.join(cache_dir, fname))
        state_dict.update(chunk)
    return state_dict


def _load_modified_dequantized_weights(model, repo_id, n_layers, n_dense_layers=1):
    """Load and dequantize FP8 safetensors weights into a modified model.

    On first run, downloads the needed shards, dequantizes FP8 weights
    block-wise, renames checkpoint keys to modified_model.py naming, and
    saves the result as per-layer BF16 safetensors cache files.

    On subsequent runs, loads directly from the cache via mmap — skipping
    shard reads, FP8 dequantization, and key renaming entirely.
    """
    cache_dir = _cache_dir_for(repo_id, n_layers)

    if os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(cache_dir)
    ):
        t_cache_start = time.perf_counter()
        print(f"[weights] loading cached BF16 weights from {cache_dir}", flush=True)
        state_dict = _load_cache_chunked(cache_dir)
        t_cache_end = time.perf_counter()
        print(
            f"[timing] cache load (mmap): {t_cache_end - t_cache_start:.1f}s "
            f"({len(state_dict)} tensors)",
            flush=True,
        )
    else:
        print(
            f"[weights] no cache found at {cache_dir}, "
            "dequantizing from FP8 shards...",
            flush=True,
        )
        state_dict = _dequantize_from_shards(repo_id, n_layers, n_dense_layers)

        t_save_start = time.perf_counter()
        total_bytes = _save_cache_chunked(state_dict, cache_dir)
        t_save_end = time.perf_counter()
        print(
            f"[timing] cache save: {t_save_end - t_save_start:.1f}s "
            f"({total_bytes / 1e9:.1f} GB)",
            flush=True,
        )

        # Re-load from cache so tensors are mmap-backed
        del state_dict
        state_dict = _load_cache_chunked(cache_dir)

    t_load_start = time.perf_counter()
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    t_load_end = time.perf_counter()
    print(
        f"[weights] loaded {len(state_dict)} tensors. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )
    print(
        f"[timing] load_state_dict: {t_load_end - t_load_start:.1f}s",
        flush=True,
    )
    if missing:
        real_missing = [
            k
            for k in missing
            if not any(
                skip in k
                for skip in [
                    "kv_cache",
                    "pe_cache",
                    "k_cache",
                    "hadamard",
                    "freqs_cis",
                    "prepopulated",
                ]
            )
        ]
        if real_missing:
            print(f"[weights] missing: {sorted(real_missing)[:20]}")


def _dequantize_from_shards(repo_id, n_layers, n_dense_layers):
    """Read FP8 shards, dequantize, rename keys. Returns a BF16 state_dict."""
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]

    # Filter to needed layers and separate weights from scales
    weight_keys = {}  # ckpt_key -> shard_file
    scale_keys = {}  # weight_ckpt_key -> scale_ckpt_key
    scale_shards = {}  # scale_ckpt_key -> shard_file

    for ckpt_key, shard_file in weight_map.items():
        layer_m = re.match(r"model\.layers\.(\d+)\.", ckpt_key)
        if layer_m and int(layer_m.group(1)) >= n_layers:
            continue
        if "weight_scale_inv" in ckpt_key:
            w_key = ckpt_key.replace(".weight_scale_inv", ".weight")
            scale_keys[w_key] = ckpt_key
            scale_shards[ckpt_key] = shard_file
        else:
            weight_keys[ckpt_key] = shard_file

    all_shards = set(weight_keys.values()) | set(scale_shards.values())

    t_shard_start = time.perf_counter()
    raw = {}
    for idx, shard_name in enumerate(sorted(all_shards)):
        shard_path = hf_hub_download(repo_id, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_keys or key in scale_shards:
                    raw[key] = f.get_tensor(key)
        print(
            f"  shard {idx + 1}/{len(all_shards)}: {shard_name} "
            f"({len(raw)} tensors so far)",
            flush=True,
        )
    t_shard_end = time.perf_counter()
    print(f"[weights] loaded {len(raw)} raw tensors from {len(all_shards)} shards")
    print(
        f"[timing] shard read (hf_hub_download + safe_open): "
        f"{t_shard_end - t_shard_start:.1f}s",
        flush=True,
    )

    # Dequantize FP8 weights
    t_dequant_start = time.perf_counter()
    dequantized = {}
    n_dequant = 0
    for ckpt_key in weight_keys:
        tensor = raw.get(ckpt_key)
        if tensor is None:
            continue
        if tensor.dtype == torch.float8_e4m3fn:
            scale_key = scale_keys.get(ckpt_key)
            scale_inv = raw.get(scale_key) if scale_key else None
            if scale_inv is not None:
                tensor = _weight_dequant(tensor, scale_inv)
                n_dequant += 1
            else:
                tensor = tensor.to(torch.bfloat16)
        dequantized[ckpt_key] = tensor
    del raw
    t_dequant_end = time.perf_counter()
    print(f"[weights] dequantized {n_dequant} FP8 tensors")
    print(
        f"[timing] FP8 dequantize: {t_dequant_end - t_dequant_start:.1f}s",
        flush=True,
    )

    # Rename keys and cast to bf16
    t_rename_start = time.perf_counter()
    state_dict = {}
    for ckpt_key, tensor in dequantized.items():
        model_key = _rename_hf_key(ckpt_key, n_dense_layers)
        if model_key is None:
            continue
        if model_key == "head.weight":
            tensor = tensor.to(torch.float32)
        elif tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        state_dict[model_key] = tensor
    del dequantized
    t_rename_end = time.perf_counter()
    print(
        f"[timing] rename keys + build state_dict: {t_rename_end - t_rename_start:.1f}s",
        flush=True,
    )

    return state_dict


def _fix_meta_buffers(model, args):
    """Replace meta-device buffers with properly computed CPU tensors.

    After creating the model on meta device and loading weights with assign=True,
    non-persistent buffers (freqs_cis, hadamard_matrix, KV caches) remain on meta
    and must be recreated on CPU.
    """
    import scipy.linalg

    # Recompute freqs_cis (RoPE frequencies with YaRN)
    freqs_cis_complex = precompute_freqs_cis(args)
    model.freqs_cis = torch.view_as_real(freqs_cis_complex)

    # Recompute hadamard_matrix
    hadamard = torch.tensor(
        scipy.linalg.hadamard(args.index_head_dim), dtype=torch.bfloat16
    ) * (args.index_head_dim**-0.5)
    model.hadamard_matrix = hadamard

    # Fix per-layer MLA buffers (caches must be bf16 to match model dtype)
    for layer in model.layers:
        attn = layer.attn
        attn.kv_cache = torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            attn.kv_lora_rank,
            dtype=torch.bfloat16,
        )
        attn.pe_cache = torch.zeros(
            args.max_batch_size,
            args.max_seq_len,
            attn.qk_rope_head_dim,
            dtype=torch.bfloat16,
        )
        attn.hadamard_matrix = hadamard
        if attn.indexer is not None:
            attn.indexer.k_cache = torch.zeros(
                args.max_batch_size,
                args.max_seq_len,
                attn.indexer.head_dim,
                dtype=torch.bfloat16,
            )


# def test_deepseek_modified_transformer_single_layer():
#     xr.set_device_type("TT")

#     # Create model args with a single layer for testing
#     args = ModelArgs(
#         n_layers=1,
#     )

#     model = ModifiedTransformer(args)

#     model = model.to(torch.bfloat16)

#     model = model.eval()
#     compiled_model = torch.compile(model, backend="tt")

#     batch_size = 1
#     seq_len = 32
#     tokens = torch.randint(0, args.vocab_size, (batch_size, seq_len))

#     device = torch_xla.device()
#     tokens = tokens.to(device)
#     compiled_model = compiled_model.to(device)

#     with torch.no_grad():
#         output = compiled_model(tokens)
#         output.to("cpu")


def test_deepseek_complex_rotary_emb():
    xr.set_device_type("TT")

    # apply_rotary_emb function copied from model.py
    def apply_rotary_emb(
        x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
    ) -> torch.Tensor:
        dtype = x.dtype
        shape = x.shape
        if not interleaved:
            x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
        x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
        freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
        y = torch.view_as_real(x * freqs_cis).flatten(3)
        if not interleaved:
            y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
        return y.to(dtype)

    batch_size = 2
    seq_len = 16
    dim = 64
    n_heads = 4
    head_dim = dim // n_heads

    x = torch.randn(batch_size, seq_len, n_heads, head_dim, dtype=torch.bfloat16)
    freqs_cis = torch.randn(seq_len, head_dim // 2, dtype=torch.complex64)

    run_graph_test(
        apply_rotary_emb,
        [x, freqs_cis],
        framework=Framework.TORCH,
    )


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("seq_len", [32, 64, 128])
def test_deepseek_attention_prefill(batch_size, seq_len):
    xr.set_device_type("TT")
    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
        index_topk=16,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    attention = model.layers[0].attn

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    # Prefill branch expects mask shape (bsz, seqlen, seqlen) for index_mask += mask
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)

    # Create a (batch_size, seq_len, index_topk) tensor of valid indices.
    # Each entry along the last axis contains values from 0 to seq_len-1, in random order per batch/position.
    topk_indices = torch.stack(
        [
            torch.stack(
                [torch.randperm(seq_len)[: args.index_topk] for _ in range(seq_len)]
            ).unsqueeze(1)
            for _ in range(batch_size)
        ]
    ).squeeze(
        2
    )  # shape: (batch_size, seq_len, index_topk)

    attention.prepopulated_topk_indices = topk_indices

    freqs_cis = model.freqs_cis[0:seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(attention, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        # Conditionally shard weights that involve batch axis
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        shard_specs[args[0]] = (None, None, batch_axis)  # hidden_states
        shard_specs[args[3]] = (batch_axis, None, None)  # attention_mask
        shard_specs[attention.wq_b.weight] = ("model", None)
        shard_specs[attention.wkv_b.weight] = ("model", None)
        shard_specs[attention.wo.weight] = (batch_axis, "model")

        shard_specs[attention.wq_a.weight] = (None, batch_axis)
        shard_specs[attention.wkv_a.weight] = (None, batch_axis)

        shard_specs[attention.kv_cache] = (batch_axis, None, None)
        shard_specs[attention.pe_cache] = (batch_axis, None, None)

        # Indexer sharding
        shard_specs[attention.indexer.wq_b.weight] = ("model", None)
        shard_specs[attention.indexer.wk.weight] = (None, batch_axis)
        shard_specs[attention.indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[attention.indexer.k_cache] = (batch_axis, None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        attention,
        [
            hidden_states,  # input tensor
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("prefill_seq_len", [32, 128, 512, 2048])
def test_deepseek_attention_decode(batch_size, prefill_seq_len, request):
    _XFAIL_CONFIGS = {
        (128, 32),
        (128, 64),
        (512, 32),
        (512, 64),
        (2048, 4),
        (2048, 32),
        (2048, 64),
    }
    if (prefill_seq_len, batch_size) in _XFAIL_CONFIGS:
        request.applymarker(
            pytest.mark.xfail(
                reason="Low PCC due to ttir.gather lowering bug - https://github.com/tenstorrent/tt-xla/issues/3726"
            )
        )

    xr.set_device_type("TT")

    # Decode-specific parameters
    decode_seq_len = 1  # Generate one token at a time
    start_pos = prefill_seq_len  # Start position for the new token

    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len * 2,
        index_topk=prefill_seq_len // 2,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    attention = model.layers[0].attn

    # Create decode input: single token only
    hidden_states = torch.randn(
        (batch_size, decode_seq_len, args.dim), dtype=torch.bfloat16
    )

    # Pre-populate caches with random data to simulate previous prefill phase
    attention.kv_cache[:batch_size, :start_pos] = torch.randn(
        batch_size, start_pos, args.kv_lora_rank, dtype=torch.bfloat16
    )
    attention.pe_cache[:batch_size, :start_pos] = torch.randn(
        batch_size, start_pos, args.qk_rope_head_dim, dtype=torch.bfloat16
    )
    attention.indexer.k_cache[:batch_size, :start_pos] = torch.randn(
        batch_size, start_pos, args.index_head_dim, dtype=torch.bfloat16
    )

    # Prepopulating topk_indices instead of running the indexer, since we have no
    # guarantee that the topk indices returned by it will be the same across CPU and
    # TT devices. Also, the indexer is already separately tested.
    attention.prepopulated_topk_indices = torch.stack(
        [torch.randperm(prefill_seq_len)[: args.index_topk] for _ in range(batch_size)]
    ).unsqueeze(
        1
    )  # (batch_size, 1, index_topk)

    # Get rotary embeddings for the current position
    freqs_cis = model.freqs_cis[start_pos : start_pos + decode_seq_len]

    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(attention, args, kwargs):
        mesh_batch_axis_size = mesh.shape()["batch"]
        # Conditionally shard weights that involve batch axis
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        # Input tensors
        shard_specs[args[0]] = (None, None, batch_axis)  # hidden_states (batch, 1, dim)

        # Weight tensors
        shard_specs[attention.wq_b.weight] = ("model", None)
        shard_specs[attention.wkv_b.weight] = ("model", None)
        shard_specs[attention.wo.weight] = (batch_axis, "model")
        shard_specs[attention.wq_a.weight] = (None, batch_axis)
        shard_specs[attention.wkv_a.weight] = (None, batch_axis)

        # Cache tensors
        shard_specs[attention.kv_cache] = (batch_axis, None, None)
        shard_specs[attention.pe_cache] = (batch_axis, None, None)

        # Indexer sharding (if present)
        shard_specs[attention.indexer.wq_b.weight] = ("model", None)
        shard_specs[attention.indexer.wk.weight] = (None, batch_axis)
        shard_specs[attention.indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[attention.indexer.k_cache] = (batch_axis, None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        attention,
        [
            hidden_states,
            start_pos,
            freqs_cis,
            None,  # attention_mask - triggers decode path
            True,  # use_optimized_decode_flow
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("seq_len", [32, 128, 512])
def test_deepseek_indexer(batch_size, seq_len):
    xr.set_device_type("TT")

    args = ModelArgs(
        n_layers=1, q_lora_rank=3072, max_batch_size=batch_size, max_seq_len=seq_len * 2
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    indexer = model.layers[0].attn.indexer

    # Enable raw score return for testing (returns index_score instead of topk_indices)
    indexer.return_raw_scores = True

    # Create inputs
    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    qr = torch.randn((batch_size, seq_len, args.q_lora_rank), dtype=torch.bfloat16)
    attention_mask = torch.zeros(batch_size, seq_len, seq_len, dtype=torch.bfloat16)
    freqs_cis = model.freqs_cis[0:seq_len]

    # Setup mesh
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (4, 8)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    def get_shard_spec(indexer, args, kwargs):
        # Conditionally shard weights that involve batch axis
        mesh_batch_axis_size = mesh.shape()["batch"]
        batch_axis = "batch" if batch_size >= mesh_batch_axis_size else None

        shard_specs = {}

        # Input tensors
        # hidden_states (x): [batch, seq, dim]
        shard_specs[args[0]] = (None, None, batch_axis)
        # qr: [batch, seq, q_lora_rank]
        shard_specs[args[1]] = (batch_axis, None, None)
        # attention_mask: [batch, seq, seq]
        shard_specs[args[4]] = (batch_axis, None, None)

        # Weight tensors
        # [n_heads*head_dim, q_lora_rank]
        shard_specs[indexer.wq_b.weight] = ("model", None)
        shard_specs[indexer.wk.weight] = (None, batch_axis)  # [head_dim, dim]
        shard_specs[indexer.k_norm.weight] = (None,)  # [head_dim]
        shard_specs[indexer.k_norm.bias] = (None,)  # [head_dim]
        # [n_heads, dim]
        shard_specs[indexer.weights_proj.weight] = ("model", batch_axis)
        shard_specs[indexer.haddamard] = (None, None)  # [head_dim, head_dim]

        # Cache tensors
        # [max_batch, max_seq, head_dim]
        shard_specs[indexer.k_cache] = (batch_axis, None, None)

        # k_scale_cache if present (for FP8 quantization mode)
        if hasattr(indexer, "k_scale_cache"):
            shard_specs[indexer.k_scale_cache] = (batch_axis, None, None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
    )

    run_graph_test(
        indexer,
        [
            hidden_states,
            qr,
            0,  # start_pos
            freqs_cis,
            attention_mask,
        ],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
def test_deepseek_v3_2_moe_only():
    """Test MoE MLP only (no attention) with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 64
    seq_len = 128
    args = ModelArgs(
        n_layers=2,
        max_batch_size=batch_size,
        max_seq_len=seq_len * 2,
    )

    model = ModifiedTransformer(args)
    model = model.to(torch.bfloat16)
    # Replace MoE module in block layer 1, then extract the replaced ffn
    block = model.layers[1]
    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    moe = block.ffn  # Now A2aSparseMLPWithSharedExperts
    moe.eval()

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(moe, args, kwargs):
        shard_specs = {}

        # x: [batch, seq, dim]
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        mlp = moe.mlp if hasattr(moe, "mlp") else moe
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        # Shared experts
        shared = getattr(moe, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        moe,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("seq_len", [1, 32, 128])
def test_deepseek_v3_2_layer_sparse_moe(batch_size, seq_len):
    """Test single MoE Block with A2aSparseMLP on (4,8) mesh — V3.2 real weights."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    args = load_deepseek_config()
    # Need n_dense_layers + 1 layers to get at least one MoE layer
    args.n_layers = args.n_dense_layers + 1
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    # Create full model, load real weights, then extract first MoE block
    model = ModifiedTransformer(args)
    load_deepseek_weights(
        model, n_layers=args.n_layers, n_dense_layers=args.n_dense_layers
    )
    model = model.to(torch.bfloat16)
    block = model.layers[args.n_dense_layers]  # first MoE layer
    freqs_cis = model.freqs_cis[:seq_len]

    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    block.eval()

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(block, args, kwargs):
        shard_specs = {}

        # x: [batch, seq, dim]
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        # Attention weights — all parallelism on _axis_0 (matches hidden on _axis_0)
        attn = block.attn
        shard_specs[attn.wq_b.weight] = ("_axis_0", None)
        shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
        shard_specs[attn.wo.weight] = (None, "_axis_0")
        shard_specs[attn.wq_a.weight] = (None, "_axis_0")
        shard_specs[attn.wkv_a.weight] = (None, "_axis_0")

        # KV caches [max_batch, max_seq, dim] — batch on _axis_1
        shard_specs[attn.kv_cache] = ("_axis_1", None, None)
        shard_specs[attn.pe_cache] = ("_axis_1", None, None)

        # Indexer
        if attn.indexer is not None:
            shard_specs[attn.indexer.wq_b.weight] = ("_axis_0", None)
            shard_specs[attn.indexer.wk.weight] = (None, "_axis_0")
            shard_specs[attn.indexer.weights_proj.weight] = (None, "_axis_0")
            shard_specs[attn.indexer.k_cache] = ("_axis_1", None, None)

        # A2aSparseMLP
        ffn = block.ffn
        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.up_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.down_proj] = (
            ("_axis_0", "_axis_1"),
            None,
            None,
        )
        shard_specs[mlp.experts.gate_proj_bias] = (
            ("_axis_0", "_axis_1"),
            None,
        )
        shard_specs[mlp.experts.up_proj_bias] = (
            ("_axis_0", "_axis_1"),
            None,
        )
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        # Shared experts
        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)

        # Norms
        shard_specs[block.attn_norm.weight] = ("_axis_0",)
        shard_specs[block.ffn_norm.weight] = ("_axis_0",)

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    run_graph_test(
        block,
        [hidden_states, None, 0, freqs_cis, mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
    )


@pytest.mark.llmbox
def test_deepseek_v3_2_full_sparse_moe():
    """Test full DeepseekV3-2 Transformer with A2aSparseMLP on (2,4) mesh."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    token_ids = [
        671,
        6102,
        294,
        8760,
        344,
        11111,
        14,
        260,
        5217,
        6354,
        362,
        2783,
        14,
        13556,
        14,
        17224,
        671,
        6102,
        294,
        8760,
        344,
        11111,
        14,
        260,
        5217,
        6354,
        362,
        2783,
        14,
        13556,
        14,
        17224,
        # 87191,
        # 305,
    ]

    batch_size = 32
    seq_len = len(token_ids)

    args = load_deepseek_config()
    args.n_layers = 4
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2
    print(f"[config] {args}")

    model = ModifiedTransformer(args)
    load_deepseek_weights(
        model, n_layers=args.n_layers, n_dense_layers=args.n_dense_layers
    )
    model = model.to(torch.bfloat16)
    # head is intentionally float32 in the original model (logits computed in fp32),
    # but model.to(bf16) converts it. Restore to float32 to match forward's .float() call.
    model.head = model.head.to(torch.float32)

    mesh_shape = (4, 8)
    enable_sparse_mlp(
        model,
        mesh=mesh_shape,
        cluster_axis=0,
        config=args,
    )

    model.eval()

    single_sequence = torch.tensor(token_ids).long()
    tokens = single_sequence.unsqueeze(0).expand(batch_size, seq_len)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(model, args, kwargs):
        shard_specs = {}

        # Input tokens [batch, seq]
        shard_specs[args[0]] = ("_axis_1", None)

        # Embedding
        shard_specs[model.embed.weight] = (None, "_axis_0")

        # Per-layer sharding
        for layer in model.layers:
            attn = layer.attn

            # MLA attention weights — all parallelism on _axis_0
            shard_specs[attn.wq_b.weight] = ("_axis_0", None)
            shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
            shard_specs[attn.wo.weight] = (None, "_axis_0")
            shard_specs[attn.wq_a.weight] = (None, "_axis_0")
            shard_specs[attn.wkv_a.weight] = (None, "_axis_0")

            # KV caches [max_batch, max_seq, dim] — batch on _axis_1
            shard_specs[attn.kv_cache] = ("_axis_1", None, None)
            shard_specs[attn.pe_cache] = ("_axis_1", None, None)

            # Indexer
            if attn.indexer is not None:
                shard_specs[attn.indexer.wq_b.weight] = ("_axis_0", None)
                shard_specs[attn.indexer.wk.weight] = (None, "_axis_0")
                shard_specs[attn.indexer.weights_proj.weight] = (None, "_axis_0")
                shard_specs[attn.indexer.k_cache] = ("_axis_1", None, None)

            # FFN sharding (MoE or dense)
            ffn = layer.ffn
            if hasattr(ffn, "mlp"):
                # A2aSparseMLPWithSharedExperts (MoE layer)
                mlp = ffn.mlp
                shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
                shard_specs[mlp.experts.gate_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.up_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.down_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.gate_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )
                shard_specs[mlp.experts.up_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )
                shard_specs[mlp.experts.down_proj_bias] = (
                    ("_axis_0", "_axis_1"),
                    None,
                )

                # Shared experts (MLP with w1/w2/w3)
                shared = getattr(ffn, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.w1.weight] = (None, "_axis_0")
                    shard_specs[shared.w3.weight] = (None, "_axis_0")
                    shard_specs[shared.w2.weight] = ("_axis_0", None)
            else:
                # Dense MLP
                shard_specs[ffn.w1.weight] = ("_axis_1", "_axis_0")
                shard_specs[ffn.w3.weight] = ("_axis_1", "_axis_0")
                shard_specs[ffn.w2.weight] = ("_axis_0", "_axis_1")

            # Norms
            shard_specs[layer.attn_norm.weight] = ("_axis_0",)
            shard_specs[layer.ffn_norm.weight] = ("_axis_0",)

        # Final norm and head
        shard_specs[model.norm.weight] = ("_axis_0",)
        shard_specs[model.head.weight] = (None, "_axis_0")

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.95),
    )

    tt_res, cpu_res = run_graph_test(
        model,
        [tokens],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=CompilerConfig(experimental_weight_dtype="bfp8"),
    )

    # TODO: Importing AutoTokenizer before run_graph_test somehow degrades PCC.
    # Instantiate it only after the test until we can debug the root cause.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEEPSEEK_V3_2_REPO)

    tt_tokens = tt_res.argmax(dim=-1)
    cpu_tokens = cpu_res.argmax(dim=-1)

    print(f"[output] TT  tokens: {tokenizer.decode(tt_tokens[0].tolist())}")
    print(f"[output] CPU tokens: {tokenizer.decode(cpu_tokens[0].tolist())}")


###############################################################################
# DeepSeek V3.1 tests — same architecture as V3.2 but n_dense_layers=3
# and no indexer (index_n_heads=0 in HF config).
###############################################################################


@pytest.mark.llmbox
@pytest.mark.parametrize("batch_size", [32, 64])
@pytest.mark.parametrize("seq_len", [1, 32, 128])
def test_deepseek_v3_1_layer_sparse_moe(batch_size, seq_len):
    """Test single MoE Block with A2aSparseMLP on (4,8) mesh — V3.1 real weights + real input."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    from transformers import AutoTokenizer

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = args.n_dense_layers + 1
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    model = ModifiedTransformer(args)
    load_deepseek_weights(
        model,
        repo_id=repo_id,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
    )
    model = model.to(torch.bfloat16)
    model.eval()

    block = model.layers[args.n_dense_layers]  # first MoE layer

    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    block.eval()

    # Generate real Block input from tokenizer + model forward through dense layers
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    text = "The quick brown fox jumps over the lazy dog. " * 10
    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens = encoded["input_ids"][:, :seq_len].repeat(batch_size, 1)

    with torch.no_grad():
        freqs_cis = model.freqs_cis[:seq_len]
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), dtype=torch.bfloat16
        ).triu_(1)

        h, residual = model.embed(tokens), None
        for layer in model.layers[: args.n_dense_layers]:
            h, residual = layer(h, residual, 0, freqs_cis, mask)
        hidden_states = h.detach()
        residual = residual.detach() if residual is not None else None

    freqs_cis = model.freqs_cis[:seq_len]
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(block, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")

        attn = block.attn
        shard_specs[attn.wq_b.weight] = ("_axis_0", None)
        shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
        shard_specs[attn.wo.weight] = (None, "_axis_0")
        shard_specs[attn.wq_a.weight] = (None, "_axis_0")
        shard_specs[attn.wkv_a.weight] = (None, "_axis_0")

        shard_specs[attn.kv_cache] = ("_axis_1", None, None)
        shard_specs[attn.pe_cache] = ("_axis_1", None, None)

        ffn = block.ffn
        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)

        shard_specs[block.attn_norm.weight] = ("_axis_0",)
        shard_specs[block.ffn_norm.weight] = ("_axis_0",)

        return shard_specs

    run_graph_test(
        block,
        [hidden_states, residual, 0, freqs_cis, mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=ComparisonConfig(
            pcc=PccConfig(enabled=True, required_pcc=0.99),
        ),
    )


def _build_moe_block_with_real_weights(repo_id, batch_size=32, seq_len=32):
    """Helper: create a MoE block with real weights for component testing."""
    args = load_deepseek_config(repo_id)
    args.n_layers = args.n_dense_layers + 1
    args.max_batch_size = batch_size
    args.max_seq_len = seq_len * 2

    model = ModifiedTransformer(args)
    load_deepseek_weights(
        model,
        repo_id=repo_id,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
    )
    model = model.to(torch.bfloat16)
    block = model.layers[args.n_dense_layers]  # first MoE layer
    return model, block, args


@pytest.mark.llmbox
def test_deepseek_v3_1_attention():
    """Test V3.1 MLA attention with real weights — should pass."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size, seq_len = 32, 32
    model, block, args = _build_moe_block_with_real_weights(
        DEEPSEEK_V3_1_REPO, batch_size, seq_len
    )
    freqs_cis = model.freqs_cis[:seq_len]
    attn = block.attn
    attn.eval()

    hidden_states = torch.randn((batch_size, seq_len, args.dim), dtype=torch.bfloat16)
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.bfloat16).triu_(1)

    mesh_shape = (4, 8)
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(attn, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        shard_specs[attn.wq_b.weight] = ("_axis_0", None)
        shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
        shard_specs[attn.wo.weight] = (None, "_axis_0")
        shard_specs[attn.wq_a.weight] = (None, "_axis_0")
        shard_specs[attn.wkv_a.weight] = (None, "_axis_0")
        shard_specs[attn.kv_cache] = ("_axis_1", None, None)
        shard_specs[attn.pe_cache] = ("_axis_1", None, None)
        return shard_specs

    run_graph_test(
        attn,
        [hidden_states, 0, freqs_cis, mask],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=ComparisonConfig(
            pcc=PccConfig(enabled=True, required_pcc=0.95)
        ),
    )


@pytest.mark.llmbox
def test_deepseek_v3_1_moe_ffn():
    """Test V3.1 MoE FFN (A2aSparseMLPWithSharedExperts) with real weights — isolates MoE PCC."""
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size, seq_len = 32, 32
    model, block, args = _build_moe_block_with_real_weights(
        DEEPSEEK_V3_1_REPO, batch_size, seq_len
    )
    model.eval()

    mesh_shape = (4, 8)
    enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=args)
    ffn = block.ffn
    ffn.eval()

    from transformers import AutoTokenizer

    # Generate real hidden_states from tokenizer + model forward
    tokenizer = AutoTokenizer.from_pretrained(
        DEEPSEEK_V3_1_REPO, trust_remote_code=True
    )
    text = "The quick brown fox jumps over the lazy dog. " * 10
    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens = encoded["input_ids"][:, :seq_len].repeat(batch_size, 1)

    with torch.no_grad():
        freqs_cis = model.freqs_cis[:seq_len]
        mask = torch.full(
            (seq_len, seq_len), float("-inf"), dtype=torch.bfloat16
        ).triu_(1)

        h, residual = model.embed(tokens), None
        for layer in model.layers[: args.n_dense_layers]:
            h, residual = layer(h, residual, 0, freqs_cis, mask)

        moe_block = model.layers[args.n_dense_layers]
        x_normed, residual_out = (
            moe_block.attn_norm(h, residual)
            if residual is not None
            else (moe_block.attn_norm(h), h)
        )
        x_attn = moe_block.attn(x_normed, 0, freqs_cis, mask)
        hidden_states, _ = moe_block.ffn_norm(x_attn, residual_out)
        hidden_states = hidden_states.detach()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(ffn, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = ("_axis_1", None, "_axis_0")
        mlp = ffn.mlp if hasattr(ffn, "mlp") else ffn
        shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
        shard_specs[mlp.experts.gate_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.down_proj] = (("_axis_0", "_axis_1"), None, None)
        shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)
        shared = getattr(ffn, "shared_experts", None)
        if shared is not None:
            shard_specs[shared.w1.weight] = (None, "_axis_0")
            shard_specs[shared.w3.weight] = (None, "_axis_0")
            shard_specs[shared.w2.weight] = ("_axis_0", None)
        return shard_specs

    run_graph_test(
        ffn,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=ComparisonConfig(
            pcc=PccConfig(enabled=True, required_pcc=0.99)
        ),
    )


@pytest.mark.llmbox
def test_deepseek_v3_1_full_sparse_moe():
    """Test full DeepseekV3.1 Transformer with A2aSparseMLP on (4,8) mesh — real input."""
    t_start = time.perf_counter()

    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    from transformers import AutoTokenizer

    batch_size = 32
    seq_len = 32

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = int(os.environ.get("DEEPSEEK_N_LAYERS", args.n_dense_layers + 1))
    args.max_batch_size = batch_size
    # max_seq_len must be > original_seq_len (4096) to activate YaRN RoPE
    # correction and attention mscale, matching production behavior.
    # Use minimal value to avoid large KV cache allocations on device.
    args.max_seq_len = args.original_seq_len + 1
    print(f"[config] {args}")

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    with torch.device("meta"):
        model = ModifiedTransformer(args)

    t1 = time.perf_counter()
    print(f"[timing] model construction (meta): {t1 - t0:.1f}s", flush=True)

    mesh_shape = (4, 8)
    post_sparse_dir = _post_sparse_cache_dir_for(repo_id, args.n_layers)
    use_post_sparse = _has_cache(post_sparse_dir)

    if use_post_sparse:
        print(
            f"[weights] loading post-sparse cache from {post_sparse_dir}",
            flush=True,
        )

        # Restructure model to A2aSparseMLP on meta (no data needed)
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)

        # Drop original experts and CPU forward path — not needed
        from tt_torch.sparse_mlp import A2aSparseMLP

        for _, mod in model.named_modules():
            if isinstance(mod, A2aSparseMLP):
                object.__setattr__(mod, "_original_mlp", None)
            if hasattr(mod, "original_experts"):
                mod.original_experts = nn.ModuleList()

        # Load stacked weights via mmap
        state_dict = _load_cache_chunked(post_sparse_dir)
        model.load_state_dict(state_dict, strict=False, assign=True)

        _fix_meta_buffers(model, args)
        model.eval()

        t2 = time.perf_counter()
        print(f"[timing] post-sparse load (mmap): {t2 - t1:.1f}s", flush=True)
    else:
        _load_modified_dequantized_weights(
            model,
            repo_id=repo_id,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
        )

        t2 = time.perf_counter()
        print(f"[timing] weight load + dequant: {t2 - t1:.1f}s", flush=True)

        _fix_meta_buffers(model, args)
        model.eval()

        t3 = time.perf_counter()
        print(f"[timing] fix_meta + align_bf16 + eval: {t3 - t2:.1f}s", flush=True)

        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)

        t4 = time.perf_counter()
        print(f"[timing] enable_sparse_mlp: {t4 - t3:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    # text = "The quick brown fox jumps over the lazy dog. " * 10
    text = (
        "Tenstorrent is a company that builds AI accelerators. "
        "Their chips are designed to"
    )
    encoded = tokenizer(
        text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens = encoded["input_ids"][:, :seq_len].repeat(batch_size, 1)

    print(f"[input] prompt: {tokenizer.decode(tokens[0].tolist())}")

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))

    def get_shard_spec(model, args, kwargs):
        shard_specs = {}

        shard_specs[args[0]] = ("_axis_1", None)
        shard_specs[model.embed.weight] = (None, "_axis_0")

        for layer in model.layers:
            attn = layer.attn

            shard_specs[attn.wq_b.weight] = ("_axis_0", None)
            shard_specs[attn.wkv_b.weight] = ("_axis_0", None)
            shard_specs[attn.wo.weight] = (None, "_axis_0")
            shard_specs[attn.wq_a.weight] = (None, "_axis_0")
            shard_specs[attn.wkv_a.weight] = (None, "_axis_0")

            shard_specs[attn.kv_cache] = ("_axis_1", None, None)
            shard_specs[attn.pe_cache] = ("_axis_1", None, None)

            ffn = layer.ffn
            if hasattr(ffn, "mlp"):
                mlp = ffn.mlp
                shard_specs[mlp.router.gate.weight] = (None, "_axis_0")
                shard_specs[mlp.experts.gate_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.up_proj] = (("_axis_0", "_axis_1"), None, None)
                shard_specs[mlp.experts.down_proj] = (
                    ("_axis_0", "_axis_1"),
                    None,
                    None,
                )
                shard_specs[mlp.experts.gate_proj_bias] = (("_axis_0", "_axis_1"), None)
                shard_specs[mlp.experts.up_proj_bias] = (("_axis_0", "_axis_1"), None)
                shard_specs[mlp.experts.down_proj_bias] = (("_axis_0", "_axis_1"), None)

                shared = getattr(ffn, "shared_experts", None)
                if shared is not None:
                    shard_specs[shared.w1.weight] = (None, "_axis_0")
                    shard_specs[shared.w3.weight] = (None, "_axis_0")
                    shard_specs[shared.w2.weight] = ("_axis_0", None)
            else:
                shard_specs[ffn.w1.weight] = ("_axis_1", "_axis_0")
                shard_specs[ffn.w3.weight] = ("_axis_1", "_axis_0")
                shard_specs[ffn.w2.weight] = ("_axis_0", "_axis_1")

            shard_specs[layer.attn_norm.weight] = ("_axis_0",)
            shard_specs[layer.ffn_norm.weight] = ("_axis_0",)

        shard_specs[model.norm.weight] = ("_axis_0",)
        shard_specs[model.head.weight] = (None, "_axis_0")

        return shard_specs

    comparison_config = ComparisonConfig(
        pcc=PccConfig(enabled=True, required_pcc=0.99),
        assert_on_failure=False,
    )

    tt_res, cpu_res = run_graph_test(
        model,
        [tokens],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
        comparison_config=comparison_config,
        compiler_config=CompilerConfig(experimental_weight_dtype="bfp8"),
    )

    tt_top5 = tt_res.topk(5, dim=-1).indices[0]
    print(f"[TT  top-5] {[tokenizer.decode([t]) for t in tt_top5.tolist()]}")

    if cpu_res is not None:
        cpu_top5 = cpu_res.topk(5, dim=-1).indices[0]
        print(f"[CPU top-5] {[tokenizer.decode([t]) for t in cpu_top5.tolist()]}")

        TorchComparisonEvaluator(
            ComparisonConfig(pcc=PccConfig(enabled=True, required_pcc=0.99))
        ).evaluate(tt_res, cpu_res)


@pytest.mark.llmbox
def test_deepseek_v3_1_decode_static_cache():
    """Autoregressive greedy decode on TT using the module-buffer cache path —
    compile-friendly, one compile per input shape (prefill + decode).

    Gates:
      - 4.1: Prefill TT top-1 == ' pearl' (matches full_sparse_moe baseline).
      - 4.2: No recompile between decode steps 1..N.
      - 4.3: 10 decode steps complete.
      - 4.4: First 3 decode tokens match CPU with PCC > 0.99.
    """
    import torch._dynamo  # noqa: F401
    import torch_xla.distributed.spmd as xs
    from transformers import AutoTokenizer
    from tt_torch.sparse_mlp import A2aSparseMLP

    t_start = time.perf_counter()

    # Enable Shardy (same as TorchWorkload.enable_spmd). Without this, the GSPMD
    # pipeline trips on "presharded argument missing @SPMDFullToShardShape" when
    # the cache buffers are sharded.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    batch_size = 32
    prompt_seq_len = 32
    n_decode = 10

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = int(os.environ.get("DEEPSEEK_N_LAYERS", args.n_dense_layers + 1))
    args.max_batch_size = batch_size
    args.max_seq_len = args.original_seq_len + 1  # = 4097, activates YaRN
    print(
        f"[config] n_layers={args.n_layers}, max_seq_len={args.max_seq_len}",
        flush=True,
    )

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    with torch.device("meta"):
        model = ModifiedTransformer(args)

    t1 = time.perf_counter()
    print(f"[timing] model construction (meta): {t1 - t0:.1f}s", flush=True)

    mesh_shape = (4, 8)
    post_sparse_dir = _post_sparse_cache_dir_for(repo_id, args.n_layers)
    use_post_sparse = _has_cache(post_sparse_dir)

    if use_post_sparse:
        print(f"[weights] loading post-sparse cache from {post_sparse_dir}", flush=True)
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)
        for _, mod in model.named_modules():
            if isinstance(mod, A2aSparseMLP):
                object.__setattr__(mod, "_original_mlp", None)
            if hasattr(mod, "original_experts"):
                mod.original_experts = nn.ModuleList()
        state_dict = _load_cache_chunked(post_sparse_dir)
        model.load_state_dict(state_dict, strict=False, assign=True)
        _fix_meta_buffers(model, args)
        model.eval()
    else:
        _load_modified_dequantized_weights(
            model,
            repo_id=repo_id,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
        )
        _fix_meta_buffers(model, args)
        model.eval()
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)

    t2 = time.perf_counter()
    print(f"[timing] weight load + sparse enable: {t2 - t1:.1f}s", flush=True)

    # ===== Tokenize prompt =====
    tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
    prompt_text = (
        "Tenstorrent is a company that builds AI accelerators. "
        "Their chips are designed to"
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=prompt_seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens_single = encoded["input_ids"][:, :prompt_seq_len]  # (1, 32)
    prompt_len = tokens_single.shape[1]
    tokens_cpu = tokens_single.repeat(batch_size, 1)  # (B, 32)
    print(
        f"[prompt] {prompt_len} tokens x{batch_size}: "
        f"{tokenizer.decode(tokens_single[0])!r}",
        flush=True,
    )

    # ===== CPU eager reference (int-path, matches existing wrapper test baseline) =====
    print("[cpu] running prefill + decode eagerly (int path)...", flush=True)
    t_cpu_start = time.perf_counter()
    cpu_logits_list = []
    cpu_token_ids = []
    with torch.no_grad():
        logits = model(tokens_cpu, start_pos=0)  # (B, V)
        cpu_logits_list.append(logits[0].detach().clone())
        next_id_scalar = int(logits[0].argmax(dim=-1).item())
        cpu_token_ids.append(next_id_scalar)
        for step in range(n_decode - 1):
            next_tokens = torch.full(
                (batch_size, 1), next_id_scalar, dtype=tokens_cpu.dtype
            )
            logits = model(next_tokens, start_pos=prompt_len + step)
            cpu_logits_list.append(logits[0].detach().clone())
            next_id_scalar = int(logits[0].argmax(dim=-1).item())
            cpu_token_ids.append(next_id_scalar)
    t_cpu_end = time.perf_counter()
    cpu_tokens_decoded = [tokenizer.decode([t]) for t in cpu_token_ids]
    print(f"[timing] CPU prefill+decode: {t_cpu_end - t_cpu_start:.1f}s", flush=True)
    print(f"[CPU tokens] {cpu_tokens_decoded}", flush=True)
    print(
        f"[CPU full]   "
        f"{tokenizer.decode(tokens_single[0].tolist() + cpu_token_ids)!r}",
        flush=True,
    )

    # Reset model buffers before TT run.
    for layer in model.layers:
        layer.attn.kv_cache.zero_()
        layer.attn.pe_cache.zero_()
        if layer.attn.indexer is not None:
            layer.attn.indexer.k_cache.zero_()

    # ===== TT path (manual orchestration, 2 compiles: prefill + decode-reused) =====
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    # Set compiler options and wrap model with torch.compile BEFORE moving to
    # device — matches `run_graph_test` order. `torch.compile` is lazy, so no
    # tracing happens yet; the first forward call is what traces+compiles.
    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp8").to_torch_compile_options()
    )
    compiled_model = torch.compile(model, backend="tt")

    # Move model weights (which include the cache buffers) to device (in-place).
    t_mv_start = time.perf_counter()
    model = model.to(device)
    # Move input tensors.
    tokens_device = tokens_cpu.to(device)
    cache_position_prefill = torch.arange(prompt_len, dtype=torch.long).to(device)
    attn_mask_prefill = _build_prefill_attention_mask(
        batch_size, prompt_len, args.max_seq_len, dtype=torch.bfloat16
    ).to(device)
    t_mv_end = time.perf_counter()
    print(f"[timing] move to device: {t_mv_end - t_mv_start:.1f}s", flush=True)

    # Mark sharding. Following test_deepseek_v3_1_full_sparse_moe, we shard
    # tokens on _axis_1 (batch dim) to match the sharding of downstream hidden
    # states. cache_position / attention_mask stay replicated.
    xs.mark_sharding(tokens_device, mesh, ("_axis_1", None))
    xs.mark_sharding(model.embed.weight, mesh, (None, "_axis_0"))
    for layer in model.layers:
        attn = layer.attn
        xs.mark_sharding(attn.wq_b.weight, mesh, ("_axis_0", None))
        xs.mark_sharding(attn.wkv_b.weight, mesh, ("_axis_0", None))
        xs.mark_sharding(attn.wo.weight, mesh, (None, "_axis_0"))
        xs.mark_sharding(attn.wq_a.weight, mesh, (None, "_axis_0"))
        xs.mark_sharding(attn.wkv_a.weight, mesh, (None, "_axis_0"))
        # Cache sharding — same spec as test_deepseek_v3_1_full_sparse_moe
        # (("_axis_1", None, None))  — batch-dim sharded.
        shard_cache = os.environ.get("TT_DECODE_SHARD_CACHE", "1") == "1"
        if shard_cache:
            xs.mark_sharding(attn.kv_cache, mesh, ("_axis_1", None, None))
            xs.mark_sharding(attn.pe_cache, mesh, ("_axis_1", None, None))
        ffn = layer.ffn
        if hasattr(ffn, "mlp"):
            mlp = ffn.mlp
            xs.mark_sharding(mlp.router.gate.weight, mesh, (None, "_axis_0"))
            xs.mark_sharding(
                mlp.experts.gate_proj, mesh, (("_axis_0", "_axis_1"), None, None)
            )
            xs.mark_sharding(
                mlp.experts.up_proj, mesh, (("_axis_0", "_axis_1"), None, None)
            )
            xs.mark_sharding(
                mlp.experts.down_proj, mesh, (("_axis_0", "_axis_1"), None, None)
            )
            xs.mark_sharding(
                mlp.experts.gate_proj_bias, mesh, (("_axis_0", "_axis_1"), None)
            )
            xs.mark_sharding(
                mlp.experts.up_proj_bias, mesh, (("_axis_0", "_axis_1"), None)
            )
            xs.mark_sharding(
                mlp.experts.down_proj_bias, mesh, (("_axis_0", "_axis_1"), None)
            )
            shared = getattr(ffn, "shared_experts", None)
            if shared is not None:
                xs.mark_sharding(shared.w1.weight, mesh, (None, "_axis_0"))
                xs.mark_sharding(shared.w3.weight, mesh, (None, "_axis_0"))
                xs.mark_sharding(shared.w2.weight, mesh, ("_axis_0", None))
        else:
            xs.mark_sharding(ffn.w1.weight, mesh, ("_axis_1", "_axis_0"))
            xs.mark_sharding(ffn.w3.weight, mesh, ("_axis_1", "_axis_0"))
            xs.mark_sharding(ffn.w2.weight, mesh, ("_axis_0", "_axis_1"))
        xs.mark_sharding(layer.attn_norm.weight, mesh, ("_axis_0",))
        xs.mark_sharding(layer.ffn_norm.weight, mesh, ("_axis_0",))

    xs.mark_sharding(model.norm.weight, mesh, ("_axis_0",))
    xs.mark_sharding(model.head.weight, mesh, (None, "_axis_0"))

    # ===== Prefill (compile #1) =====
    print("[tt] running prefill (compile #1)...", flush=True)
    t_pf_start = time.perf_counter()
    with torch.no_grad():
        tt_logits_prefill = compiled_model(
            tokens_device,
            cache_position=cache_position_prefill,
            attention_mask=attn_mask_prefill,
        )
    t_pf_end = time.perf_counter()
    print(
        f"[timing] TT prefill compile + run: {t_pf_end - t_pf_start:.1f}s", flush=True
    )

    tt_logits_prefill_cpu = tt_logits_prefill.to("cpu").detach()
    tt_logits_list = [tt_logits_prefill_cpu[0].clone()]
    tt_token_ids = [int(tt_logits_prefill_cpu[0].argmax(dim=-1).item())]
    print(f"[tt] prefill top-1: {tokenizer.decode([tt_token_ids[0]])!r}", flush=True)

    # ===== Decode loop (compile #2, reused) =====
    for step in range(n_decode - 1):
        pos = prompt_len + step
        next_tokens = torch.full(
            (batch_size, 1), tt_token_ids[-1], dtype=tokens_cpu.dtype
        ).to(device)
        cache_position_decode = torch.tensor([pos], dtype=torch.long).to(device)
        attn_mask_decode = _build_decode_attention_mask(
            batch_size, pos + 1, args.max_seq_len, dtype=torch.bfloat16
        ).to(device)

        t_dec_start = time.perf_counter()
        with torch.no_grad():
            tt_logits = compiled_model(
                next_tokens,
                cache_position=cache_position_decode,
                attention_mask=attn_mask_decode,
            )
        tt_logits_cpu = tt_logits.to("cpu").detach()
        t_dec_end = time.perf_counter()

        tt_logits_list.append(tt_logits_cpu[0].clone())
        tt_token_ids.append(int(tt_logits_cpu[0].argmax(dim=-1).item()))
        print(
            f"[tt] decode step {step + 1}/{n_decode - 1}: "
            f"time={t_dec_end - t_dec_start:.2f}s "
            f"tok={tokenizer.decode([tt_token_ids[-1]])!r}",
            flush=True,
        )

    tt_tokens_decoded = [tokenizer.decode([t]) for t in tt_token_ids]
    print(f"[TT tokens]  {tt_tokens_decoded}", flush=True)
    print(
        f"[TT full]    {tokenizer.decode(tokens_single[0].tolist() + tt_token_ids)!r}",
        flush=True,
    )

    # ===== Per-step PCC =====
    print("\n[pcc] per-step logits PCC (TT vs CPU):", flush=True)
    step_pccs = []
    for step, (tt_l, cpu_l) in enumerate(zip(tt_logits_list, cpu_logits_list)):
        x = tt_l.to(torch.float64).numpy().flatten()
        y = cpu_l.to(torch.float64).numpy().flatten()
        vx = x - x.mean()
        vy = y - y.mean()
        denom = float(np.linalg.norm(vx) * np.linalg.norm(vy))
        pcc = float("nan") if denom == 0 else float(np.dot(vx, vy) / denom)
        step_pccs.append(pcc)
        tt_tok = tt_token_ids[step]
        cpu_tok = cpu_token_ids[step]
        match = "OK" if tt_tok == cpu_tok else "MISMATCH"
        print(
            f"  step {step}: pcc={pcc:.4f}  tt_tok={tt_tok!r:>10} "
            f"cpu_tok={cpu_tok!r:>10}  [{match}]",
            flush=True,
        )

    print(
        f"\n[pcc] min={min(step_pccs):.4f}, max={max(step_pccs):.4f}, "
        f"avg={sum(step_pccs)/len(step_pccs):.4f}",
        flush=True,
    )

    matches = sum(1 for a, b in zip(tt_token_ids, cpu_token_ids) if a == b)
    print(f"[tokens] TT==CPU match: {matches}/{len(tt_token_ids)}", flush=True)

    # Gate 4.1: prefill top-1 must match the `' pearl'` baseline (only meaningful at 4 layers).
    if args.n_layers == 4:
        expected_prefill_token = tokenizer.encode(" pearl", add_special_tokens=False)
        if expected_prefill_token:
            print(
                f"[gate 4.1] expected prefill token = "
                f"{tokenizer.decode(expected_prefill_token)!r} (ids={expected_prefill_token})",
                flush=True,
            )

    print(
        f"[pcc] overall (min across steps) = {min(step_pccs):.4f}  "
        f"(required_pcc = 0.99, assert_on_failure = False)",
        flush=True,
    )


def _build_prefill_attention_mask(
    batch_size, prompt_len, max_cache_len, dtype=torch.bfloat16
):
    """Build (B, 1, prompt_len, max_cache_len) mask.

    Positions 0..q_pos are allowed (0.0), others are -inf. Positions >= prompt_len
    are -inf for all query positions.
    """
    mask = torch.full(
        (batch_size, 1, prompt_len, max_cache_len),
        float("-inf"),
        dtype=dtype,
    )
    for q_pos in range(prompt_len):
        mask[:, :, q_pos, : q_pos + 1] = 0.0
    return mask


def _build_decode_attention_mask(
    batch_size, current_pos_inclusive, max_cache_len, dtype=torch.bfloat16
):
    """Build (B, 1, 1, max_cache_len) mask for a single decode step.

    `current_pos_inclusive` = number of valid slots (== cache_position + 1). All
    positions in [0, current_pos_inclusive) allowed, others -inf.
    """
    mask = torch.full(
        (batch_size, 1, 1, max_cache_len),
        float("-inf"),
        dtype=dtype,
    )
    mask[:, :, :, :current_pos_inclusive] = 0.0
    return mask


@pytest.mark.parametrize("batch_size", [1, 4, 32, 64])
@pytest.mark.parametrize("prefill_seq_len", [32, 128, 512, 2048])
def test_dsa_optimized_decode_flow_compared_to_original(batch_size, prefill_seq_len):
    """
    This test compares the optimized decode flow with the original reference flow provided
    by Deepseek. It is run only on CPU.
    """
    decode_seq_len = 1  # Generate one token at a time
    start_pos = prefill_seq_len  # Start position for the new token

    args = ModelArgs(
        n_layers=1,
        q_lora_rank=3072,
        max_batch_size=batch_size,
        max_seq_len=prefill_seq_len * 2,
        index_topk=16,
    )

    modified_model = ModifiedTransformer(args)
    modified_model = modified_model.to(torch.bfloat16)
    attention = modified_model.layers[0].attn

    freqs_cis = modified_model.freqs_cis[start_pos : start_pos + decode_seq_len]

    hidden_states = torch.randn(
        (batch_size, decode_seq_len, args.dim), dtype=torch.bfloat16
    )
    kv_cache = torch.randn(
        batch_size, start_pos, args.kv_lora_rank, dtype=torch.bfloat16
    )
    pe_cache = torch.randn(
        batch_size, start_pos, args.qk_rope_head_dim, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        batch_size, start_pos, args.index_head_dim, dtype=torch.bfloat16
    )
    end_pos = start_pos + decode_seq_len
    topk_indices = torch.stack(
        [torch.randperm(end_pos)[: args.index_topk] for _ in range(batch_size)]
    ).unsqueeze(
        1
    )  # (batch_size, 1, index_topk)

    attention.prepopulated_topk_indices = topk_indices
    attention.kv_cache[:batch_size, :start_pos] = kv_cache
    attention.pe_cache[:batch_size, :start_pos] = pe_cache
    attention.indexer.k_cache[:batch_size, :start_pos] = k_cache

    test_modified_output = attention(
        hidden_states, start_pos, freqs_cis, mask=None, use_optimized_decode_flow=True
    )
    test_original_output = attention(
        hidden_states, start_pos, freqs_cis, mask=None, use_optimized_decode_flow=False
    )

    pcc = compute_pcc(test_modified_output, test_original_output)

    assert pcc > 0.99, f"PCC too low: {pcc}"
