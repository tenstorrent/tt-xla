# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek v3.1 tests on TT (Galaxy 4x8).

Two end-to-end tests, both hand-orchestrated (no dependency on
``run_graph_test``), so they can build on top of ``origin/main`` without
tester-infrastructure changes:

- ``test_deepseek_v3_1_full_sparse_moe``: prefill-only, A2aSparseMLP, legacy
  int-path cache, CPU PCC comparison.
- ``test_deepseek_v3_1_decode_static_cache``: autoregressive greedy decode
  through MLACache (``cache_position`` + ``attention_mask``), 2 compiles
  (prefill + decode), per-step PCC against a CPU int-path reference.

Both share the weight-loading / dequantization / sparse-cache infrastructure
below.
"""
import json
import os
import re
import time

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from huggingface_hub import hf_hub_download
from modified_model import ModelArgs
from modified_model import Transformer as ModifiedTransformer
from modified_model import precompute_freqs_cis
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file
from torch import nn
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import A2aSparseMLP, enable_sparse_mlp

from tests.infra.testers.compiler_config import CompilerConfig

DEEPSEEK_V3_1_REPO = "deepseek-ai/DeepSeek-V3.1"

FP8_BLOCK_SIZE = 128


# ---------------------------------------------------------------------------
# Weight loading / dequantization / cache
# ---------------------------------------------------------------------------


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


def load_deepseek_config(repo_id=DEEPSEEK_V3_1_REPO):
    """Download and parse the HuggingFace config.json into ModelArgs fields."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        hf_cfg = json.load(f)
    rope_scaling = hf_cfg.get("rope_scaling", {})
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


def _cache_dir_for(repo_id, n_layers):
    """Per-model BF16 safetensors cache directory (shared with build_weight_cache.py)."""
    repo_slug = repo_id.replace("/", "--")
    base = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return os.path.join(base, "tt_xla_dequant_cache", f"{repo_slug}_{n_layers}layers")


def _post_sparse_cache_dir_for(repo_id, n_layers):
    return _cache_dir_for(repo_id, n_layers) + "_post_sparse"


def _has_cache(cache_dir):
    return os.path.isdir(cache_dir) and any(
        f.endswith(".safetensors") for f in os.listdir(cache_dir)
    )


def _save_cache_chunked(state_dict, cache_dir):
    """Save state_dict as per-layer + shared safetensors chunks."""
    os.makedirs(cache_dir, exist_ok=True)
    layer_dicts = {}
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
        del layer_dicts[layer_idx]
        print(f"  [cache] saved layer {layer_idx} ({sz / 1e9:.1f} GB)", flush=True)
    return total_bytes


def _load_cache_chunked(cache_dir):
    """Load all chunk files via mmap. Returns merged state_dict."""
    state_dict = {}
    for fname in sorted(os.listdir(cache_dir)):
        if not fname.endswith(".safetensors"):
            continue
        chunk = safetensors_load_file(os.path.join(cache_dir, fname))
        state_dict.update(chunk)
    return state_dict


def _dequantize_from_shards(repo_id, n_layers, n_dense_layers):
    """Read FP8 shards, dequantize, rename keys. Returns a BF16 state_dict."""
    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    weight_keys = {}
    scale_keys = {}
    scale_shards = {}
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
    print(f"[weights] dequantized {n_dequant} FP8 tensors")

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
    return state_dict


def _load_modified_dequantized_weights(model, repo_id, n_layers, n_dense_layers=1):
    """Dequantize FP8 safetensors -> BF16, cache them, load into ``model``."""
    cache_dir = _cache_dir_for(repo_id, n_layers)
    if _has_cache(cache_dir):
        t0 = time.perf_counter()
        print(f"[weights] loading cached BF16 weights from {cache_dir}", flush=True)
        state_dict = _load_cache_chunked(cache_dir)
        t1 = time.perf_counter()
        print(
            f"[timing] cache load (mmap): {t1 - t0:.1f}s ({len(state_dict)} tensors)",
            flush=True,
        )
    else:
        print(
            f"[weights] no cache at {cache_dir} — dequantizing from FP8...", flush=True
        )
        state_dict = _dequantize_from_shards(repo_id, n_layers, n_dense_layers)
        _save_cache_chunked(state_dict, cache_dir)
        del state_dict
        state_dict = _load_cache_chunked(cache_dir)

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
    )


def _fix_meta_buffers(model, args):
    """Replace meta-device buffers with properly-computed CPU tensors.

    After meta construction + ``load_state_dict(assign=True)``, non-persistent
    buffers remain on meta. Recompute them on CPU.
    """
    import scipy.linalg

    freqs_cis_complex = precompute_freqs_cis(args)
    model.freqs_cis = torch.view_as_real(freqs_cis_complex)
    hadamard = torch.tensor(
        scipy.linalg.hadamard(args.index_head_dim), dtype=torch.bfloat16
    ) * (args.index_head_dim**-0.5)
    model.hadamard_matrix = hadamard
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


def _build_prefill_attention_mask(
    batch_size,
    prompt_len,
    max_cache_len,
    token_attention_mask=None,
    dtype=torch.bfloat16,
):
    """(B, 1, prompt_len, max_cache_len) additive causal mask.

    If ``token_attention_mask`` (shape ``(B, prompt_len)``, 1=real, 0=pad) is
    given, also masks out padding slots in the key dimension so queries don't
    attend to left-pad tokens. The tokenizer runs with ``padding_side="left"``,
    so those slots are at the front of the sequence.
    """
    mask = torch.full(
        (batch_size, 1, prompt_len, max_cache_len), float("-inf"), dtype=dtype
    )
    for q_pos in range(prompt_len):
        mask[:, :, q_pos, : q_pos + 1] = 0.0
    if token_attention_mask is not None:
        tam = token_attention_mask.to(dtype=dtype)  # (B, prompt_len)
        pad_k = torch.where(
            tam > 0,
            torch.zeros((), dtype=dtype),
            torch.full((), float("-inf"), dtype=dtype),
        )  # 0 at real tokens, -inf at pad
        mask[:, :, :, :prompt_len] = (
            mask[:, :, :, :prompt_len] + pad_k[:, None, None, :]
        )
    return mask


def _build_decode_attention_mask(
    batch_size,
    current_pos_inclusive,
    max_cache_len,
    token_attention_mask=None,
    prompt_len=None,
    dtype=torch.bfloat16,
):
    """(B, 1, 1, max_cache_len) additive mask for a single decode step.

    If ``token_attention_mask`` + ``prompt_len`` are given, also masks cache
    slots ``0..prompt_len-1`` where the prompt was padding, so the decode
    step doesn't attend to left-pad KV entries that prefill wrote in.
    """
    mask = torch.full((batch_size, 1, 1, max_cache_len), float("-inf"), dtype=dtype)
    mask[:, :, :, :current_pos_inclusive] = 0.0
    if token_attention_mask is not None and prompt_len is not None:
        tam = token_attention_mask.to(dtype=dtype)  # (B, prompt_len)
        pad_k = torch.where(
            tam > 0,
            torch.zeros((), dtype=dtype),
            torch.full((), float("-inf"), dtype=dtype),
        )
        mask[:, :, :, :prompt_len] = (
            mask[:, :, :, :prompt_len] + pad_k[:, None, None, :]
        )
    return mask


def _apply_shard_specs(model, mesh, args, batch_size, tokens_device):
    """Apply mark_sharding to all weights + the input tokens."""
    xs.mark_sharding(tokens_device, mesh, ("_axis_1", None))
    xs.mark_sharding(model.embed.weight, mesh, (None, "_axis_0"))
    for layer in model.layers:
        attn = layer.attn
        xs.mark_sharding(attn.wq_b.weight, mesh, ("_axis_0", None))
        xs.mark_sharding(attn.wkv_b.weight, mesh, ("_axis_0", None))
        xs.mark_sharding(attn.wo.weight, mesh, (None, "_axis_0"))
        xs.mark_sharding(attn.wq_a.weight, mesh, (None, "_axis_0"))
        xs.mark_sharding(attn.wkv_a.weight, mesh, (None, "_axis_0"))
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
            # Biases are None for DeepSeek v3.1 (no-bias MoE) — skip.
            if mlp.experts.gate_proj_bias is not None:
                xs.mark_sharding(
                    mlp.experts.gate_proj_bias,
                    mesh,
                    (("_axis_0", "_axis_1"), None),
                )
            if mlp.experts.up_proj_bias is not None:
                xs.mark_sharding(
                    mlp.experts.up_proj_bias, mesh, (("_axis_0", "_axis_1"), None)
                )
            if mlp.experts.down_proj_bias is not None:
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


def _build_and_load_model(args, repo_id, prefer_cpu_capable=False):
    """Build a meta-device ModifiedTransformer and populate real weights.

    Two weight-load paths with different trade-offs. The caller picks via
    ``prefer_cpu_capable``; whether to actually *run* a CPU reference is a
    separate concern gated at the test level.

    - Post-sparse cache (``prefer_cpu_capable=False``, default — matches the
      old-branch speed profile): loads pre-stacked expert weights via mmap,
      then nulls ``_original_mlp`` + ``original_experts`` to free memory.
      ``enable_sparse_mlp`` runs on meta, so no runtime expert stacking.
      CPU forward through the sparse layer will fail — do not call
      ``model(tokens_cpu, ...)`` after this path.
    - Dequantized cache (``prefer_cpu_capable=True``): loads the per-expert
      BF16 cache, then ``enable_sparse_mlp`` runtime-stacks experts while
      keeping the original MoE module alive via ``cpu_forward_module`` —
      CPU forward works. For large n_layers this is *much* slower (both
      the stacking step and the subsequent move-to-device).

    If the preferred path's cache is missing, falls back to the other
    (logging). If neither cache exists, dequantizes from FP8 shards.

    Returns ``(model, mesh_shape)``.
    """
    with torch.device("meta"):
        model = ModifiedTransformer(args)
    mesh_shape = (4, 8)
    post_sparse_dir = _post_sparse_cache_dir_for(repo_id, args.n_layers)
    dequant_dir = _cache_dir_for(repo_id, args.n_layers)

    def _load_post_sparse():
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

    def _load_dequant():
        print(f"[weights] loading dequantized cache from {dequant_dir}", flush=True)
        _load_modified_dequantized_weights(
            model,
            repo_id=repo_id,
            n_layers=args.n_layers,
            n_dense_layers=args.n_dense_layers,
        )
        _fix_meta_buffers(model, args)
        model.eval()
        enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=0, config=args)

    if prefer_cpu_capable:
        if _has_cache(dequant_dir):
            _load_dequant()
            return model, mesh_shape
        if _has_cache(post_sparse_dir):
            print(
                "[weights] prefer_cpu_capable=True but dequant cache missing; "
                "falling back to post-sparse cache (CPU forward will be broken)",
                flush=True,
            )
            _load_post_sparse()
            return model, mesh_shape
    else:
        if _has_cache(post_sparse_dir):
            _load_post_sparse()
            return model, mesh_shape
        if _has_cache(dequant_dir):
            print(
                "[weights] post-sparse cache missing; falling back to "
                "dequant cache (slower — runtime expert stacking)",
                flush=True,
            )
            _load_dequant()
            return model, mesh_shape

    # Neither cache exists — dequantize from FP8 shards (populates dequant cache).
    _load_dequant()
    return model, mesh_shape


def _reset_caches(model):
    for layer in model.layers:
        layer.attn.kv_cache.zero_()
        layer.attn.pe_cache.zero_()
        if layer.attn.indexer is not None:
            layer.attn.indexer.k_cache.zero_()


def _pcc(a, b):
    x = a.to(torch.float64).flatten().numpy()
    y = b.to(torch.float64).flatten().numpy()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = float(np.linalg.norm(vx) * np.linalg.norm(vy))
    return float("nan") if denom == 0 else float(np.dot(vx, vy) / denom)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_deepseek_v3_1_full_sparse_moe():
    """Prefill-only forward on the (4,8) mesh with A2aSparseMLP. Real input,
    real weights, CPU PCC comparison."""
    t_start = time.perf_counter()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    from transformers import AutoTokenizer

    batch_size = 32
    seq_len = 32

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = int(os.environ.get("DEEPSEEK_N_LAYERS", args.n_dense_layers + 1))
    args.max_batch_size = batch_size
    # max_seq_len must exceed original_seq_len (4096) to activate YaRN mscale.
    args.max_seq_len = args.original_seq_len + 1
    print(
        f"[config] n_layers={args.n_layers}, max_seq_len={args.max_seq_len}", flush=True
    )

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    # DEEPSEEK_CPU_REFERENCE=1 runs the CPU eager reference and reports PCC.
    # Requires the dequant cache (which keeps the original MoE alive for CPU
    # forward); post-sparse cache is incompatible. Disabled by default so
    # large-layer runs use the fast post-sparse path.
    run_cpu_reference = os.environ.get("DEEPSEEK_CPU_REFERENCE", "0") == "1"

    model, mesh_shape = _build_and_load_model(
        args, repo_id, prefer_cpu_capable=run_cpu_reference
    )

    t1 = time.perf_counter()
    print(f"[timing] model build + weight load: {t1 - t0:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, add_bos_token=True, padding_side="left"
    )
    prompt_text = (
        "Tenstorrent is a company that builds AI accelerators. "
        "Their chips are designed to"
    )
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens_single = encoded["input_ids"][:, :seq_len]
    tokens_cpu = tokens_single.repeat(batch_size, 1)
    print(
        f"[input] prompt: {tokenizer.decode(tokens_single[0].tolist())!r}", flush=True
    )

    # ===== CPU reference (opt-in via DEEPSEEK_CPU_REFERENCE=1) =====
    cpu_logits = None
    if run_cpu_reference:
        print("[cpu] running prefill eagerly (int path)...", flush=True)
        t_cpu_start = time.perf_counter()
        with torch.no_grad():
            cpu_logits = model(tokens_cpu, start_pos=0)
        t_cpu_end = time.perf_counter()
        cpu_top5 = cpu_logits[0].topk(5, dim=-1).indices
        print(f"[timing] CPU prefill: {t_cpu_end - t_cpu_start:.1f}s", flush=True)
        print(
            f"[CPU top-5] {[tokenizer.decode([t]) for t in cpu_top5.tolist()]}",
            flush=True,
        )
        _reset_caches(model)
    else:
        print(
            "[cpu] SKIPPED — set DEEPSEEK_CPU_REFERENCE=1 to run CPU golden",
            flush=True,
        )

    # ===== TT path =====
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp_bf8").to_torch_compile_options()
    )
    compiled_model = torch.compile(model, backend="tt")

    t_mv_start = time.perf_counter()
    model = model.to(device)
    tokens_device = tokens_cpu.to(device)
    t_mv_end = time.perf_counter()
    print(f"[timing] move to device: {t_mv_end - t_mv_start:.1f}s", flush=True)

    _apply_shard_specs(model, mesh, args, batch_size, tokens_device)

    print("[tt] running prefill...", flush=True)
    t_tt_start = time.perf_counter()
    with torch.no_grad():
        tt_logits = compiled_model(tokens_device, start_pos=0)
    tt_logits_cpu = tt_logits.to("cpu").detach()
    t_tt_end = time.perf_counter()
    print(
        f"[timing] TT prefill (compile + run): {t_tt_end - t_tt_start:.1f}s", flush=True
    )

    tt_top5 = tt_logits_cpu[0].topk(5, dim=-1).indices
    print(
        f"[TT  top-5] {[tokenizer.decode([t]) for t in tt_top5.tolist()]}", flush=True
    )

    if cpu_logits is not None:
        pcc = _pcc(tt_logits_cpu, cpu_logits.detach())
        print(f"[pcc] logits PCC (TT vs CPU) = {pcc:.4f} (required: 0.99)", flush=True)
    else:
        print("[pcc] SKIPPED — no CPU reference", flush=True)


def test_deepseek_v3_1_decode_static_cache():
    """Autoregressive greedy decode on TT using the MLACache path.

    Two compiles: prefill (prompt_seq_len), decode (seq_len=1, reused).
    """
    import torch._dynamo  # noqa: F401

    t_start = time.perf_counter()

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.set_device_type("TT")
    torch_xla.runtime.use_spmd()

    from transformers import AutoTokenizer

    batch_size = 32
    prompt_seq_len = 32
    n_decode = 10

    repo_id = DEEPSEEK_V3_1_REPO
    args = load_deepseek_config(repo_id)
    args.n_layers = int(os.environ.get("DEEPSEEK_N_LAYERS", args.n_dense_layers + 1))
    args.max_batch_size = batch_size
    args.max_seq_len = args.original_seq_len + 1  # = 4097, activates YaRN
    print(
        f"[config] n_layers={args.n_layers}, max_seq_len={args.max_seq_len}", flush=True
    )

    t0 = time.perf_counter()
    print(f"[timing] config + init: {t0 - t_start:.1f}s", flush=True)

    # DEEPSEEK_CPU_REFERENCE=1 runs the CPU eager reference (prefill + decode
    # loop) and reports per-step PCC. Requires the dequant cache; post-sparse
    # cache is incompatible. Disabled by default so 61-layer runs use the fast
    # post-sparse path.
    run_cpu_reference = os.environ.get("DEEPSEEK_CPU_REFERENCE", "0") == "1"

    model, mesh_shape = _build_and_load_model(
        args, repo_id, prefer_cpu_capable=run_cpu_reference
    )

    t1 = time.perf_counter()
    print(f"[timing] model build + weight load: {t1 - t0:.1f}s", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, trust_remote_code=True, add_bos_token=True, padding_side="left"
    )
    prompt_text = "Tell me a short story."
    encoded = tokenizer(
        prompt_text,
        return_tensors="pt",
        max_length=prompt_seq_len,
        truncation=True,
        padding="max_length",
    )
    tokens_single = encoded["input_ids"][:, :prompt_seq_len]
    prompt_len = tokens_single.shape[1]
    tokens_cpu = tokens_single.repeat(batch_size, 1)
    # Tokenizer attention_mask: 1 for real tokens, 0 for left-pad. Used below
    # to mask pad slots out of prefill/decode attention (pad=<eos>, so without
    # this queries silently attend to pad KV entries).
    token_attn_mask_cpu = encoded["attention_mask"][:, :prompt_seq_len].repeat(
        batch_size, 1
    )
    print(
        f"[prompt] {prompt_len} tokens x{batch_size}: "
        f"{tokenizer.decode(tokens_single[0])!r}",
        flush=True,
    )

    # ===== CPU eager reference (opt-in via DEEPSEEK_CPU_REFERENCE=1) =====
    cpu_logits_list = []
    cpu_token_ids = []
    if run_cpu_reference:
        print("[cpu] running prefill + decode eagerly (int path)...", flush=True)
        t_cpu_start = time.perf_counter()
        with torch.no_grad():
            logits = model(tokens_cpu, start_pos=0)
            cpu_logits_list.append(logits[0].detach().clone())
            next_id = int(logits[0].argmax(dim=-1).item())
            cpu_token_ids.append(next_id)
            for step in range(n_decode - 1):
                next_tokens = torch.full(
                    (batch_size, 1), next_id, dtype=tokens_cpu.dtype
                )
                logits = model(next_tokens, start_pos=prompt_len + step)
                cpu_logits_list.append(logits[0].detach().clone())
                next_id = int(logits[0].argmax(dim=-1).item())
                cpu_token_ids.append(next_id)
        t_cpu_end = time.perf_counter()
        print(
            f"[timing] CPU prefill+decode: {t_cpu_end - t_cpu_start:.1f}s",
            flush=True,
        )
        print(
            f"[CPU tokens] {[tokenizer.decode([t]) for t in cpu_token_ids]}",
            flush=True,
        )
        print(
            f"[CPU full]   "
            f"{tokenizer.decode(tokens_single[0].tolist() + cpu_token_ids)!r}",
            flush=True,
        )
        _reset_caches(model)
    else:
        print(
            "[cpu] SKIPPED — set DEEPSEEK_CPU_REFERENCE=1 to run CPU golden",
            flush=True,
        )

    # ===== TT path =====
    torch._dynamo.reset()

    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(
        CompilerConfig(experimental_weight_dtype="bfp_bf8").to_torch_compile_options()
    )
    compiled_model = torch.compile(model, backend="tt")

    t_mv_start = time.perf_counter()
    model = model.to(device)
    tokens_device = tokens_cpu.to(device)
    cache_position_prefill = torch.arange(prompt_len, dtype=torch.long).to(device)
    attn_mask_prefill = _build_prefill_attention_mask(
        batch_size,
        prompt_len,
        args.max_seq_len,
        token_attention_mask=token_attn_mask_cpu,
        dtype=torch.bfloat16,
    ).to(device)
    t_mv_end = time.perf_counter()
    print(f"[timing] move to device: {t_mv_end - t_mv_start:.1f}s", flush=True)

    _apply_shard_specs(model, mesh, args, batch_size, tokens_device)

    # ===== Prefill =====
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
        f"[timing] TT prefill compile + run: {t_pf_end - t_pf_start:.1f}s",
        flush=True,
    )
    tt_logits_prefill_cpu = tt_logits_prefill.to("cpu").detach()
    tt_logits_list = [tt_logits_prefill_cpu[0].clone()]
    tt_token_ids = [int(tt_logits_prefill_cpu[0].argmax(dim=-1).item())]
    print(
        f"[tt] prefill top-1: {tokenizer.decode([tt_token_ids[0]])!r}",
        flush=True,
    )

    # ===== Decode loop =====
    for step in range(n_decode - 1):
        pos = prompt_len + step
        next_tokens = torch.full(
            (batch_size, 1), tt_token_ids[-1], dtype=tokens_cpu.dtype
        ).to(device)
        cache_position_decode = torch.tensor([pos], dtype=torch.long).to(device)
        attn_mask_decode = _build_decode_attention_mask(
            batch_size,
            pos + 1,
            args.max_seq_len,
            token_attention_mask=token_attn_mask_cpu,
            prompt_len=prompt_len,
            dtype=torch.bfloat16,
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

    print(
        f"[TT tokens]  {[tokenizer.decode([t]) for t in tt_token_ids]}",
        flush=True,
    )
    print(
        f"[TT full]    "
        f"{tokenizer.decode(tokens_single[0].tolist() + tt_token_ids)!r}",
        flush=True,
    )

    if cpu_logits_list:
        print("\n[pcc] per-step logits PCC (TT vs CPU):", flush=True)
        step_pccs = []
        for step, (tt_l, cpu_l) in enumerate(zip(tt_logits_list, cpu_logits_list)):
            pcc = _pcc(tt_l, cpu_l)
            step_pccs.append(pcc)
            tt_tok = tt_token_ids[step]
            cpu_tok = cpu_token_ids[step]
            match = "OK" if tt_tok == cpu_tok else "MISMATCH"
            print(
                f"  step {step}: pcc={pcc:.4f}  "
                f"tt_tok={tt_tok!r:>10} cpu_tok={cpu_tok!r:>10}  [{match}]",
                flush=True,
            )
        print(
            f"\n[pcc] min={min(step_pccs):.4f}, max={max(step_pccs):.4f}, "
            f"avg={sum(step_pccs) / len(step_pccs):.4f}",
            flush=True,
        )
        matches = sum(1 for a, b in zip(tt_token_ids, cpu_token_ids) if a == b)
        print(
            f"[tokens] TT==CPU match: {matches}/{len(tt_token_ids)}",
            flush=True,
        )
    else:
        print("\n[pcc] SKIPPED — no CPU reference", flush=True)
