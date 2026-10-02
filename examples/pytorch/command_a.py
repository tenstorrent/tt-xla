# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Command-A Reasoning (111B, Cohere2) bring-up on Tenstorrent hardware.
# Uses SPMD tensor parallelism across available TT devices.
# Supports incremental testing with --num-layers.

import argparse
import os
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from tt_torch.sharding import sharding_constraint_hook

MODEL_NAME = "CohereLabs/command-a-reasoning-08-2025"


def setup_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def create_mesh():
    n = xr.global_runtime_device_count()
    mesh = Mesh(np.array(range(n)), (1, n), ("batch", "model"))
    print(f"Mesh: (1, {n})")
    return mesh


def load_model(num_layers=None):
    """Load Command-A with optional layer count override."""
    config = AutoConfig.from_pretrained(MODEL_NAME)
    if num_layers is not None:
        config.num_hidden_layers = num_layers
    # Force all layers to full_attention to avoid StaticSlidingWindowLayer
    # (keeps sliding_window value for mask creation but no layer uses it)
    if hasattr(config, "layer_types"):
        config.layer_types = ["full_attention"] * config.num_hidden_layers

    print(f"Loading {MODEL_NAME} ({config.num_hidden_layers} layers)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loaded: {config.num_hidden_layers}L, {config.hidden_size}H, "
          f"{config.num_attention_heads}QH/{config.num_key_value_heads}KVH")
    return model, tokenizer, config


def mark_sharding(model, cache, mesh):
    """Column-parallel QKV/gate/up, row-parallel O/down (per bring-up guide §6)."""
    # KV cache
    for layer in cache.layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    # Model weights
    for layer in model.model.layers:
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

    # All-gather logits from lm_head
    hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
    model.lm_head.register_forward_hook(hook)


def build_inputs(prompt_text, tokenizer, config, max_cache_len):
    inputs = tokenizer(prompt_text, return_tensors="pt", return_attention_mask=True)
    prompt_len = inputs.input_ids.shape[1]

    cache = StaticCache(
        config=config, max_batch_size=1, max_cache_len=max_cache_len,
        device="cpu", dtype=torch.bfloat16,
    )
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    cache.early_initialization(
        batch_size=1, num_heads=kv_heads, head_dim=head_dim,
        dtype=torch.bfloat16, device="cpu",
    )

    attn_mask = torch.ones((1, max_cache_len), dtype=torch.long)
    attn_mask[:, :prompt_len] = inputs.attention_mask

    return {
        "input_ids": inputs.input_ids,
        "past_key_values": cache,
        "cache_position": torch.arange(0, prompt_len),
        "use_cache": True,
        "attention_mask": attn_mask,
    }, prompt_len, cache


def to_device(model, input_args, device):
    for l in input_args["past_key_values"].layers:
        l.keys = l.keys.to(device)
        l.values = l.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)
    return model, input_args


def cpu_reference(model, tokenizer, config, prompt_text, max_cache_len):
    """Run on CPU and return top-5 logits for comparison."""
    inputs = tokenizer(prompt_text, return_tensors="pt")
    prompt_len = inputs.input_ids.shape[1]
    cache = StaticCache(
        config=config, max_batch_size=1, max_cache_len=max_cache_len,
        device="cpu", dtype=torch.bfloat16,
    )
    kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    cache.early_initialization(
        batch_size=1, num_heads=kv_heads, head_dim=head_dim,
        dtype=torch.bfloat16, device="cpu",
    )
    with torch.no_grad():
        out = model(inputs.input_ids, past_key_values=cache,
                    cache_position=torch.arange(prompt_len), use_cache=True)
    logits = out.logits[0, -1]
    top = torch.topk(logits.float(), 5)
    return logits, [tokenizer.decode(i) for i in top.indices.tolist()], \
           [f"{v:.1f}" for v in top.values.tolist()]


def main():
    parser = argparse.ArgumentParser(description="Command-A on TT hardware")
    parser.add_argument("--num-layers", type=int, default=None,
                        help="Override number of layers (for incremental testing)")
    parser.add_argument("--max-cache-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--bfp8", action="store_true", help="Enable bfp8 weight conversion")
    parser.add_argument("--bfp4", action="store_true", help="Enable bfp_bf4 weight conversion")
    parser.add_argument("--validate-only", action="store_true",
                        help="Just compare CPU vs TT logits, no generation")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt (non-interactive)")
    args = parser.parse_args()

    # --- Phase 1: Load model ---
    model_tt, tokenizer, config = load_model(args.num_layers)

    # --- Phase 2: TT setup ---
    print("\n--- TT Setup ---")
    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()
    mesh = create_mesh()

    # Set compile options
    compile_options = {}
    if args.bfp4:
        compile_options["experimental_weight_dtype"] = "bfp4"
        print("BFP4 weight conversion enabled")
    elif args.bfp8:
        compile_options["experimental_weight_dtype"] = "bfp8"
        print("BFP8 weight conversion enabled")
    if compile_options:
        torch_xla.set_custom_compile_options(compile_options)

    # --- Phase 3: Warmup compile with a short prompt ---
    warmup_prompt = "Hello"
    input_args, prompt_len, cache = build_inputs(
        warmup_prompt, tokenizer, config, args.max_cache_len)
    model_tt, input_args = to_device(model_tt, input_args, device)
    mark_sharding(model_tt, cache, mesh)

    print("Compiling (first run, please wait)...")
    compiled = torch.compile(model_tt, backend="tt")
    t0 = time.perf_counter()
    with torch.no_grad():
        compiled(**input_args)
    t1 = time.perf_counter()
    print(f"Compiled in {t1-t0:.1f}s. Ready!\n")

    # --- Phase 4: Interactive loop ---
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            break
        if user_input.strip().lower() in ("quit", "quit()", "exit", "q"):
            break
        if not user_input.strip():
            continue

        # Simple instruction format (full chat template is ~1200 tokens which
        # degrades quality at bfp4). This short format fits in small cache.
        prompt_text = f"User: {user_input}\nAssistant:"

        input_args, prompt_len, cache = build_inputs(
            prompt_text, tokenizer, config, args.max_cache_len)
        for l in cache.layers:
            l.keys = l.keys.to(device)
            l.values = l.values.to(device)
        input_args["input_ids"] = input_args["input_ids"].to(device)
        input_args["cache_position"] = input_args["cache_position"].to(device)
        input_args["attention_mask"] = input_args["attention_mask"].to(device)
        mark_sharding(model_tt, cache, mesh)

        generated_ids = []
        max_new = min(args.max_new_tokens, args.max_cache_len - prompt_len)
        if max_new <= 0:
            print("(prompt too long for cache)\n")
            continue

        print("Assistant: ", end="", flush=True)
        prefill_start = time.perf_counter()
        decode_start = None
        token_count = 0

        with torch.no_grad():
            for step in range(max_new):
                out = compiled(**input_args)
                logits = out.logits[:, -1].to("cpu")

                if step == 0:
                    prefill_end = time.perf_counter()
                    decode_start = time.perf_counter()

                # Repetition penalty
                for tid in set(generated_ids):
                    if logits[0, tid] > 0:
                        logits[0, tid] /= args.repetition_penalty
                    else:
                        logits[0, tid] *= args.repetition_penalty

                # Sample
                if args.temperature <= 0:
                    next_id = logits.argmax(dim=-1)
                else:
                    scaled = logits / args.temperature
                    sorted_l, sorted_i = torch.sort(scaled, descending=True, dim=-1)
                    cum = torch.cumsum(F.softmax(sorted_l, dim=-1), dim=-1)
                    mask = cum - F.softmax(sorted_l, dim=-1) >= args.top_p
                    sorted_l[mask] = float("-inf")
                    probs = F.softmax(
                        torch.zeros_like(scaled).scatter(-1, sorted_i, sorted_l), dim=-1
                    )
                    next_id = torch.multinomial(probs, 1).squeeze(-1)

                token_id = next_id.item()
                generated_ids.append(token_id)
                if token_id == tokenizer.eos_token_id:
                    break
                print(tokenizer.decode(next_id), end="", flush=True)
                token_count += 1

                input_args["input_ids"] = next_id.unsqueeze(0).to(device)
                pos = input_args["cache_position"].to("cpu")
                input_args["cache_position"] = torch.tensor([pos[-1:] + 1]).to(device)

        decode_end = time.perf_counter()
        print()
        if decode_start:
            prefill_time = prefill_end - prefill_start
            decode_time = decode_end - decode_start
            tps = token_count / decode_time if decode_time > 0 else 0
            print(f"--- {token_count} tokens | prefill {prefill_time:.1f}s | "
                  f"decode {tps:.2f} tok/s | total {decode_end - prefill_start:.1f}s ---\n")


if __name__ == "__main__":
    main()
