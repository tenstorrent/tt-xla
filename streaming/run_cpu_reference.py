"""CPU reference run — produces a per-layer activation golden trace.

Architecture mirrors run_layer_stream.py but runs entirely on CPU
in eager mode (no torch.compile, no XLA). Same layer-streaming pattern:

  - Top-level (embed/head/norm/hc_*) loaded once on CPU.
  - For each token step, for each layer_id:
      1. Build fresh model instance with HF weights for that layer.
      2. enable_sparse_mlp + strip on CPU.
      3. Forward (eager) — block(h, sp, input_ids).
      4. Save h_out to disk via pcc_utils.capture_or_compare.
      5. Drop instance.

Heavy: MoE on CPU with 256 experts × 6 active is slow. Use small
NUM_LAYERS / BATCH for quick verification. Or run overnight for full.

Env vars:
    STREAM_NUM_LAYERS, STREAM_BATCH_SIZE, STREAM_PROMPT_LEN,
    STREAM_MAX_NEW_TOKENS, STREAM_REF_DIR (where activations save).

Run:
    source venv/activate
    STREAM_REF_MODE=capture STREAM_REF_DIR=/tmp/golden \\
    STREAM_NUM_LAYERS=4 STREAM_BATCH_SIZE=32 STREAM_PROMPT_LEN=16 \\
    STREAM_MAX_NEW_TOKENS=2 \\
    python streaming/run_cpu_reference.py
"""
from __future__ import annotations

import gc
import os
import time
from typing import List

import psutil
import torch
from torch import nn

from tt_torch.sparse_mlp import enable_sparse_mlp
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import _strip_cpu_golden_refs
from streaming.pcc_utils import capture_or_compare, REF_MODE
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)


PROMPT_LEN = int(os.environ.get("STREAM_PROMPT_LEN", "16"))
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "1"))
BATCH_SIZE = int(os.environ.get("STREAM_BATCH_SIZE", "32"))
NUM_LAYERS = int(os.environ.get("STREAM_NUM_LAYERS", "4"))


def _malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _log(tag):
    _malloc_trim()
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    print(f"[{tag:38s}] rss={rss:6.2f} GB", flush=True)


def _build_skeleton(args):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)


def _build_layer_instance(layer_id, args):
    """Build a CPU model instance with layer_id's HF weights loaded
    and sparse-MLP rewritten."""
    m = _build_skeleton(args)
    block_sd = weight_loader.load_block_state_dict(layer_id)
    prefix = f"layers.{layer_id}."
    stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in block_sd.items()}
    m.layers[layer_id].load_state_dict(stripped, strict=False)
    del block_sd, stripped
    gc.collect()
    enable_sparse_mlp(m.layers[layer_id], mesh=(1, 1), cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(m.layers[layer_id])
    return m


def main():
    if REF_MODE != "capture":
        print(f"WARNING: STREAM_REF_MODE={REF_MODE!r} (expected 'capture'). "
              f"This script is intended for capturing reference activations.",
              flush=True)

    torch.manual_seed(0)
    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = BATCH_SIZE
    args.n_layers = NUM_LAYERS
    args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    args.max_seq_len = max(((PROMPT_LEN + MAX_NEW_TOKENS + 31) // 32) * 32, max_cr)
    print(f"[args] layers={NUM_LAYERS} bsz={BATCH_SIZE} prompt={PROMPT_LEN} "
          f"new={MAX_NEW_TOKENS} max_seq_len={args.max_seq_len}", flush=True)

    _log("baseline")

    # ---- Build primary model (will hold top-level + persistent kv_cache) ----
    primary = _build_skeleton(args)
    embed_sd = weight_loader.load_embed_state_dict()
    primary.embed.load_state_dict(embed_sd, strict=False)
    del embed_sd
    top_sd = weight_loader.load_top_level_state_dict()
    primary.load_state_dict(top_sd, strict=False)
    del top_sd
    gc.collect()
    _log("post-top-level")

    # ---- Initialize per-layer kv_cache buffers in primary by loading
    # layer 0's HF weights once + sparse_mlp on each layer slot, then
    # zero out parameters but keep buffers. This mirrors v5's persistent
    # kv_cache init step. ----
    print("\n[init] per-layer setup (sparse_mlp + buffers stay) ...", flush=True)
    for layer_id in range(NUM_LAYERS):
        # Load layer 0 weights as placeholder (kv_caches are zero-init
        # regardless of which weights we use).
        block_sd_0 = weight_loader.load_block_state_dict(0)
        prefix = "layers.0."
        stripped0 = {(k[len(prefix):] if k.startswith(prefix) else k): v
                     for k, v in block_sd_0.items()}
        primary.layers[layer_id].load_state_dict(stripped0, strict=False)
        del block_sd_0, stripped0
        gc.collect()
        enable_sparse_mlp(primary.layers[layer_id], mesh=(1, 1),
                         cluster_axis=0, config=args, verbose=False)
        _strip_cpu_golden_refs(primary.layers[layer_id])
        # Drop the placeholder weights — keep only buffers.
        for sub in primary.layers[layer_id].modules():
            for name in list(sub._parameters.keys()):
                sub._parameters[name] = None
        gc.collect()
        _log(f"post-init-l{layer_id}")

    # ---- Tokenize ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    PROMPTS = ["How are you today?", "What is the capital of France?"] + [f"Prompt {i}" for i in range(2, BATCH_SIZE)]
    PROMPTS = PROMPTS[:BATCH_SIZE]
    rows = []
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.shape[0] >= PROMPT_LEN:
            ids = ids[-PROMPT_LEN:]
        else:
            pad = torch.full((PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long)
            ids = torch.cat([pad, ids], dim=0)
        rows.append(ids)
    prompt_ids = torch.stack(rows, dim=0).contiguous()

    # ---- Inference (eager, CPU) ----
    print("\n[infer] CPU reference run ...", flush=True)

    generated: List[List[int]] = [[] for _ in range(BATCH_SIZE)]
    next_input_ids = prompt_ids
    sp_value = PROMPT_LEN

    for step in range(MAX_NEW_TOKENS):
        t_step = time.time()
        sp = torch.tensor(sp_value if step == 0 else (PROMPT_LEN + step - 1),
                          dtype=torch.long)

        with torch.inference_mode():
            # ---- Embed ----
            h = primary.embed(next_input_ids)
            h = h.unsqueeze(2).repeat(1, 1, primary.hc_mult, 1)
            capture_or_compare(step, -1, h, tag="embed")
            _log(f"s{step} post-embed")

            # ---- Layer iteration ----
            for layer_id in range(NUM_LAYERS):
                t_layer = time.time()
                # Build temp instance with this layer's weights.
                inst = _build_layer_instance(layer_id, args)
                # Splice persistent kv_cache buffers from primary (preserve state).
                temp_block = inst.layers[layer_id]
                primary_block = primary.layers[layer_id]
                # Walk both module trees in parallel; copy primary's
                # buffers into temp so kv_cache state propagates.
                temp_subs = list(temp_block.modules())
                prim_subs = list(primary_block.modules())
                for tsub, psub in zip(temp_subs, prim_subs):
                    for name, pbuf in list(psub._buffers.items()):
                        if pbuf is None:
                            continue
                        tsub._buffers[name] = pbuf

                # Eager forward.
                h_out = temp_block(h, sp, next_input_ids)
                # Capture activation.
                capture_or_compare(step, layer_id, h_out, tag="out")
                # Update primary's buffers from temp (kv_cache mutated).
                for tsub, psub in zip(temp_subs, prim_subs):
                    for name in list(psub._buffers.keys()):
                        psub._buffers[name] = tsub._buffers.get(name)

                del inst, temp_block, h
                gc.collect()
                h = h_out
                t_total = time.time() - t_layer
                cr = args.compress_ratios[layer_id]
                rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
                print(f"[cpu s{step} l{layer_id:2d} cr={cr:3d}] {t_total:.1f}s rss={rss:.2f} GB", flush=True)

            # ---- Head ----
            logits = primary.head(h, primary.hc_head_fn, primary.hc_head_scale,
                                   primary.hc_head_base, primary.norm)
            capture_or_compare(step, NUM_LAYERS, logits, tag="head")
            next_ids = logits.detach().argmax(dim=-1)
            for i in range(BATCH_SIZE):
                generated[i].append(int(next_ids[i].item()))
            next_input_ids = next_ids.unsqueeze(1)
            del h, logits
            gc.collect()
            print(f"[cpu s{step}] total {time.time()-t_step:.1f}s ids[:4]={next_ids[:4].tolist()}", flush=True)

    print("\n[done] decoded:", flush=True)
    for i, (prompt, ids) in enumerate(zip(PROMPTS[:4], generated[:4])):
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  [{i}] {prompt!r} -> {cont!r}", flush=True)


if __name__ == "__main__":
    main()
