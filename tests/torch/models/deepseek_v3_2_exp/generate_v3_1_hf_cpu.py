#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Ground-truth CPU inference of DeepSeek V3.1 using the HuggingFace model.

Creates the model via AutoModelForCausalLM.from_config, then loads and
dequantizes FP8 safetensors weights manually. This is necessary because
transformers' built-in FineGrainedFP8Config(dequantize=True) doesn't handle
MoE expert weights (they stay as raw FP8) or dimensions not divisible by the
block size (kv_a_proj_with_mqa is 576x7168).

Usage:
    python generate_v3_1_hf_cpu.py [--max-tokens N] [--prompt "text"] [--n-layers N]
"""

import argparse
import json
import re
import sys
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

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


def load_dequantized_weights(model, n_layers):
    """Load safetensors from HF cache, dequantize FP8 weights, load into model.

    Manual dequantization is necessary because transformers' built-in FP8 quantizer
    (FineGrainedFP8HfQuantizer) dequantizes on the fly during forward using Triton/CUDA
    kernels, which are not available on CPU.  We pre-dequantize all FP8 weights to bf16
    so the model can run with standard CPU ops.
    """
    index_path = hf_hub_download(REPO_ID, "model.safetensors.index.json")
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

    # Dequantize FP8 tensors: multiply each 128x128 block by its scale factor
    state_dict = {}
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
        state_dict[ckpt_key] = tensor
    print(f"[weights] dequantized {n_dequant} FP8 tensors", flush=True)

    # Cast to bf16. Keep e_score_correction_bias in fp32 —
    # BF16 truncation at magnitude ~5 causes ~0.015 error which flips MoE routing.
    for key, tensor in state_dict.items():
        if "e_score_correction_bias" not in key and tensor.dtype != torch.bfloat16:
            state_dict[key] = tensor.to(torch.bfloat16)

    # assign=True replaces meta tensors with real data directly
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    print(
        f"[weights] loaded {len(state_dict)} tensors. "
        f"Missing: {len(missing)}, Unexpected: {len(unexpected)}",
        flush=True,
    )
    if missing:
        real_missing = [k for k in missing if "weight_scale_inv" not in k]
        if real_missing:
            print(f"[weights] missing: {sorted(real_missing)[:10]}", flush=True)


def generate(model, tokenizer, prompt, max_new_tokens=10, max_prompt_len=32):
    """Autoregressive greedy generation."""
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=max_prompt_len,
        truncation=True,
    )
    tokens = encoded["input_ids"][:, :max_prompt_len]
    seq_len = tokens.shape[1]
    generated_ids = tokens[0].tolist()

    print(f"\n{'='*60}", flush=True)
    print(f"Prompt ({seq_len} tokens): {tokenizer.decode(tokens[0])!r}", flush=True)
    print(f"Generating {max_new_tokens} tokens...", flush=True)
    print(f"{'='*60}\n", flush=True)
    print(tokenizer.decode(tokens[0]), end="", flush=True)

    # Prefill
    t0 = time.time()
    with torch.no_grad():
        out = model(tokens, use_cache=True)
        logits = out.logits[:, -1, :].float()
        past_kv = out.past_key_values
    prefill_time = time.time() - t0

    next_id = logits[0].argmax().item()
    generated_ids.append(next_id)
    print(tokenizer.decode([next_id]), end="", flush=True)
    print(f"\n  [prefill: {prefill_time:.1f}s]", file=sys.stderr, flush=True)

    # Decode
    for step in range(1, max_new_tokens):
        t0 = time.time()
        with torch.no_grad():
            out = model(
                torch.tensor([[next_id]]),
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = out.logits[:, -1, :].float()
            past_kv = out.past_key_values
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
        description="Ground-truth DeepSeek V3.1 generation on CPU (HF model)"
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument("--max-prompt-len", type=int, default=32)
    parser.add_argument(
        "--n-layers", type=int, default=None, help="Number of layers (default: all 61)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog. " * 10,
    )
    args = parser.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    print("[tokenizer] Loading...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    print("[config] Loading...", flush=True)
    config = AutoConfig.from_pretrained(REPO_ID)
    if hasattr(config, "quantization_config"):
        del config.quantization_config
    if args.n_layers is not None:
        config.num_hidden_layers = args.n_layers
    print(f"[config] num_hidden_layers={config.num_hidden_layers}", flush=True)

    print("[model] Creating from config (meta tensors)...", flush=True)
    t0 = time.time()
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)
    print(f"[model] Created in {time.time() - t0:.1f}s", flush=True)

    print("[weights] Loading and dequantizing FP8 weights...", flush=True)
    t0 = time.time()
    load_dequantized_weights(model, config.num_hidden_layers)
    # After load_state_dict(assign=True), non-persistent buffers like
    # rotary_emb.inv_freq remain on meta device.  Re-create the rotary
    # embedding on CPU so inv_freq is computed from the config (not zeroed).
    with torch.device("cpu"):
        model.model.rotary_emb = type(model.model.rotary_emb)(config=config)
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
    )


if __name__ == "__main__":
    main()
