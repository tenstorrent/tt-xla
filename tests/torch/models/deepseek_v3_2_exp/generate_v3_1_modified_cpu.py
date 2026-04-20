#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ground-truth CPU inference of DeepSeek V3.1 using the modified model.

Creates the model from modified_model.py, then loads and dequantizes FP8
safetensors weights manually. This confirms that the modified model produces
sensible tokens, matching the HF reference implementation.

Usage:
    python generate_v3_1_modified_cpu.py [--max-tokens N] [--prompt "text"] [--n-layers N]
"""

import argparse
import json
import math
import os
import re
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

# Add this directory to path for modified_model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from modified_model import ModelArgs, Transformer, precompute_freqs_cis

REPO_ID = "deepseek-ai/DeepSeek-V3.1"
BLOCK_SIZE = 128


def _weight_dequant(weight, scale_inv, block_size=BLOCK_SIZE):
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


def load_config(repo_id, n_layers=None, max_batch_size=1, max_seq_len=None):
    """Load HF config.json and create ModelArgs."""
    config_path = hf_hub_download(repo_id, "config.json")
    with open(config_path) as f:
        hf_cfg = json.load(f)

    rope_scaling = hf_cfg.get("rope_scaling", {})
    original_seq_len = rope_scaling.get("original_max_position_embeddings", 4096)

    args = ModelArgs(
        max_batch_size=max_batch_size,
        # max_seq_len must exceed original_seq_len to activate YaRN
        max_seq_len=max_seq_len or original_seq_len + 1,
        vocab_size=hf_cfg["vocab_size"],
        dim=hf_cfg["hidden_size"],
        inter_dim=hf_cfg["intermediate_size"],
        moe_inter_dim=hf_cfg["moe_intermediate_size"],
        n_layers=n_layers or hf_cfg["num_hidden_layers"],
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
        original_seq_len=original_seq_len,
        rope_theta=hf_cfg.get("rope_theta", 10000.0),
        rope_factor=rope_scaling.get("factor", 40),
        beta_fast=rope_scaling.get("beta_fast", 32),
        beta_slow=rope_scaling.get("beta_slow", 1),
        mscale=rope_scaling.get("mscale", 1.0),
        index_n_heads=hf_cfg.get("index_n_heads", 0),
        index_head_dim=hf_cfg.get("index_head_dim", 128),
        index_topk=hf_cfg.get("index_topk", 2048),
    )
    return args


def load_dequantized_weights(model, args):
    """Load safetensors from HF, dequantize FP8 weights, rename keys, load into model.

    Manual dequantization is necessary because transformers' built-in FP8 quantizer
    uses Triton/CUDA kernels not available on CPU.
    """
    index_path = hf_hub_download(REPO_ID, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    weight_map = index["weight_map"]
    n_layers = args.n_layers

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
    shard_list = sorted(all_shards)

    raw = {}
    for idx, shard_name in enumerate(shard_list):
        t0 = time.time()
        shard_path = hf_hub_download(REPO_ID, shard_name)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in weight_keys or key in scale_shards:
                    raw[key] = f.get_tensor(key)
        elapsed = time.time() - t0
        print(
            f"  shard {idx + 1}/{len(shard_list)}: {shard_name} "
            f"({elapsed:.1f}s, {len(raw)} tensors so far)",
            flush=True,
        )

    # Dequantize FP8 tensors
    n_dequant = 0
    dequantized = {}
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
    print(f"[weights] dequantized {n_dequant} FP8 tensors", flush=True)

    # Rename keys from HF format to modified model format.
    # Cast to bf16 except:
    #   - gate.bias (e_score_correction_bias): BF16 truncation flips MoE routing
    #   - head.weight (lm_head): model calls F.linear(x.float(), weight) which
    #     requires matching dtypes
    state_dict = {}
    for ckpt_key, tensor in dequantized.items():
        model_key = _rename_hf_key(ckpt_key, args.n_dense_layers)
        if model_key is None:
            continue
        if model_key == "head.weight":
            tensor = tensor.to(torch.float32)
        elif "gate.bias" not in model_key and tensor.dtype != torch.bfloat16:
            tensor = tensor.to(torch.bfloat16)
        state_dict[model_key] = tensor

    # assign=True replaces meta tensors with real data directly
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}",
        flush=True,
    )
    if missing:
        # Filter out expected missing keys (non-persistent buffers, caches)
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
            print(f"[weights] missing: {sorted(real_missing)[:20]}", flush=True)


def fix_meta_buffers(model, args):
    """Replace meta device buffers with properly computed CPU tensors.

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

    # Fix per-layer MLA buffers.
    # The model was designed with torch.set_default_dtype(torch.bfloat16), so
    # caches must be bf16 to match the dtype of values stored during forward.
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


def generate(model, tokenizer, prompt, max_new_tokens=10, max_prompt_len=32, pad=False):
    """Autoregressive greedy generation using the modified model."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_prompt_len,
        truncation=True,
        padding="max_length" if pad else False,
    )
    tokens = encoded["input_ids"][:, :max_prompt_len]
    seq_len = tokens.shape[1]
    generated_ids = tokens[0].tolist()

    print(f"\n{'='*60}", flush=True)
    print(f"Prompt ({seq_len} tokens): {tokenizer.decode(tokens[0])!r}", flush=True)
    print(f"Generating {max_new_tokens} tokens...", flush=True)
    print(f"{'='*60}\n", flush=True)
    print(tokenizer.decode(tokens[0]), end="", flush=True)

    # Prefill: process all input tokens at once
    t0 = time.time()
    with torch.no_grad():
        logits = model(tokens, start_pos=0)
    prefill_time = time.time() - t0

    next_id = logits[0].argmax().item()
    generated_ids.append(next_id)
    print(tokenizer.decode([next_id]), end="", flush=True)
    print(f"\n  [prefill: {prefill_time:.1f}s]", file=sys.stderr, flush=True)

    # Decode: generate one token at a time using KV cache
    for step in range(1, max_new_tokens):
        t0 = time.time()
        with torch.no_grad():
            logits = model(
                torch.tensor([[next_id]]),
                start_pos=seq_len + step - 1,
            )
        decode_time = time.time() - t0

        next_id = logits[0].argmax().item()
        generated_ids.append(next_id)
        print(tokenizer.decode([next_id]), end="", flush=True)
        print(
            f"\n  [step {step + 1}/{max_new_tokens}: {decode_time:.1f}s, token={next_id}]",
            file=sys.stderr,
            flush=True,
        )
        if next_id == tokenizer.eos_token_id:
            break

    full_text = tokenizer.decode(generated_ids)
    print(f"\n\n{'='*60}", flush=True)
    print(f"Full output: {full_text!r}", flush=True)
    print(f"{'='*60}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Ground-truth DeepSeek V3.1 generation on CPU (modified model)"
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--max-prompt-len", type=int, default=32)
    parser.add_argument(
        "--pad",
        action="store_true",
        help="Pad prompt to max-prompt-len (matches the decode test setup).",
    )
    parser.add_argument(
        "--n-layers", type=int, default=None, help="Number of layers (default: all 61)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Tenstorrent is a company that builds AI accelerators. "
        "Their chips are designed to",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print("[tokenizer] Loading...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    print("[config] Loading...", flush=True)
    model_args = load_config(REPO_ID, n_layers=args.n_layers, max_batch_size=1)
    print(
        f"[config] n_layers={model_args.n_layers}, "
        f"max_seq_len={model_args.max_seq_len}, "
        f"original_seq_len={model_args.original_seq_len}",
        flush=True,
    )

    print("[model] Creating from ModelArgs (meta tensors)...", flush=True)
    t0 = time.time()
    with torch.device("meta"):
        model = Transformer(model_args)
    print(f"[model] Created in {time.time() - t0:.1f}s", flush=True)

    print("[weights] Loading and dequantizing FP8 weights...", flush=True)
    t0 = time.time()
    load_dequantized_weights(model, model_args)
    fix_meta_buffers(model, model_args)

    # Verify no meta tensors remain
    meta_tensors = [
        name for name, p in model.named_parameters() if p.device.type == "meta"
    ] + [name for name, b in model.named_buffers() if b.device.type == "meta"]
    assert not meta_tensors, f"Meta tensors remain after loading: {meta_tensors}"

    model.eval()
    print(f"[weights] Done in {time.time() - t0:.1f}s", flush=True)

    generate(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        max_prompt_len=args.max_prompt_len,
        pad=args.pad,
    )


if __name__ == "__main__":
    main()
