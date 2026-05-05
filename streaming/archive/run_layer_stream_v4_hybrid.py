"""[ARCHIVED] Layer-streaming inference v4 — persistent skeleton + weights swap + del+round-trip release.

Superseded by ../run_layer_stream.py. See archive/README.md for context.

Combines v2 (persistent skeleton for kv_cache) with v3's release
mechanism (del temp instance + round-trip h):

  - Persistent `skeleton` Transformer with kv_cache buffers shipped
    once at init and shared across all token steps. Provides correct
    KV-cache state for autoregressive decode.
  - For each layer iteration:
      1. Build TEMP model instance with this layer's HF weights on
         CPU + sparse-MLP rewritten.
      2. Ship temp's _parameters into skeleton.layers[layer_id] —
         this REPLACES the previous _parameters, dropping refs to
         old XLA tensors → device buffers freed.
      3. run_block(skeleton.layers[layer_id], h, sp, ids) → h_out
      4. Round-trip h_out via host to break IR chain back to this
         layer's weights.
      5. del temp, h_out, h (input). gc.collect + sync.
      6. h = h_next.

Why this works: device DRAM is bounded by one layer's weights at any
time because old _parameters get replaced + the temp instance is del'd
(no closure ref). KV-cache buffers stay on skeleton and persist.
"""
from __future__ import annotations

import gc
import os
import time
from typing import List

import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch import nn
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.sparse_mlp import enable_sparse_mlp
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
    _top_level_shard_spec, _upload_with_sharding,
)
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)


PROMPT_LEN = int(os.environ.get("STREAM_PROMPT_LEN", "16"))
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "2"))
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
    sys_used = psutil.virtual_memory().used / 1e9
    print(f"[{tag:38s}] rss={rss:6.2f} sys={sys_used:6.2f} GB", flush=True)


def _build_skeleton(args):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)


def _ship_top_level(model, mesh, device):
    embed_sd = weight_loader.load_embed_state_dict()
    model.embed.load_state_dict(embed_sd, strict=False)
    del embed_sd
    top_sd = weight_loader.load_top_level_state_dict()
    model.load_state_dict(top_sd, strict=False)
    del top_sd
    gc.collect()
    top_specs = _top_level_shard_spec(model)
    top_specs_by_id = {id(t): ps for t, ps in top_specs.items()}
    del top_specs
    _ship_module_handle_path(model.embed, top_specs_by_id, mesh, device,
                             verbose=False, tag="top:embed")
    _ship_module_handle_path(model.norm, top_specs_by_id, mesh, device,
                             verbose=False, tag="top:norm")
    _ship_module_handle_path(model.head, top_specs_by_id, mesh, device,
                             verbose=False, tag="top:head")
    for pname in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        p = model._parameters.get(pname)
        if p is None or p.device.type != "cpu":
            continue
        partition_spec = top_specs_by_id.get(id(p))
        xla_t = _upload_with_sharding(p.data.detach(), mesh, partition_spec, device)
        model._parameters[pname] = nn.Parameter(xla_t, requires_grad=False)
    torch_xla.sync(wait=True)
    gc.collect()


def _build_temp_layer(layer_id, args, mesh_shape):
    """Build a CPU-only model instance with layer_id's weights loaded
    + sparse-MLP rewritten. Returns a tuple (temp_model, block_ref)."""
    m = _build_skeleton(args)
    block_sd = weight_loader.load_block_state_dict(layer_id)
    prefix = f"layers.{layer_id}."
    stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in block_sd.items()}
    m.layers[layer_id].load_state_dict(stripped, strict=False)
    del block_sd, stripped
    gc.collect()
    enable_sparse_mlp(m.layers[layer_id], mesh=mesh_shape, cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(m.layers[layer_id])
    return m


def _ship_temp_into_skeleton(temp_block, skel_block, mesh, device):
    """For each parameter in temp_block, upload to device and replace
    skel_block's _parameters slot. Buffers (kv_cache!) on skel are
    untouched. Old XLA tensors in skel slots are released as their refs
    drop.
    """
    specs = _block_shard_spec(skel_block, mesh)
    specs_by_id = {id(t): ps for t, ps in specs.items()}
    del specs
    temp_subs = list(temp_block.modules())
    skel_subs = list(skel_block.modules())
    assert len(temp_subs) == len(skel_subs), \
        f"module count mismatch temp={len(temp_subs)} skel={len(skel_subs)}"
    n = 0
    for tsub, ssub in zip(temp_subs, skel_subs):
        for name, tp in list(tsub._parameters.items()):
            if tp is None:
                continue
            cur = ssub._parameters.get(name)
            partition_spec = (specs_by_id.get(id(cur))
                              if cur is not None else None)
            xla_t = _upload_with_sharding(tp.data.detach(), mesh, partition_spec, device)
            ssub._parameters[name] = nn.Parameter(xla_t, requires_grad=False)
            n += 1
    torch_xla.sync(wait=True)
    return n


def _round_trip(t, device):
    """Force materialization, drop IR chain. Round-trip via host."""
    cpu = t.detach().to("cpu")
    return cpu.to(device)


def main():
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    n = xr.global_runtime_device_count()
    mesh_shape = (2, 4) if n == 8 else (1, n)
    mesh = Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

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

    # ---- Persistent skeleton: top-level + per-layer kv_cache buffers ----
    skeleton = _build_skeleton(args)
    _ship_top_level(skeleton, mesh, device)
    _log("post-top-level")

    # Initialize each skeleton layer with layer 0's weights to populate
    # the StackedExperts module structure + ship buffers (kv_cache,
    # freqs_cis). The actual weights will be replaced per-iteration.
    print("\n[init] per-layer setup (sparse_mlp + kv_cache buffers) ...", flush=True)
    for layer_id in range(NUM_LAYERS):
        block_sd_0 = weight_loader.load_block_state_dict(0)
        prefix = "layers.0."
        stripped0 = {(k[len(prefix):] if k.startswith(prefix) else k): v
                     for k, v in block_sd_0.items()}
        skeleton.layers[layer_id].load_state_dict(stripped0, strict=False)
        del block_sd_0, stripped0
        gc.collect()
        enable_sparse_mlp(skeleton.layers[layer_id], mesh=mesh_shape,
                         cluster_axis=0, config=args, verbose=False)
        _strip_cpu_golden_refs(skeleton.layers[layer_id])
        gc.collect()
        # Ship initial weights + kv_cache buffers (kv_cache stays on
        # device; weights will be replaced on first layer execute).
        block_specs = _block_shard_spec(skeleton.layers[layer_id], mesh)
        block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
        del block_specs
        _ship_module_handle_path(skeleton.layers[layer_id], block_specs_by_id,
                                 mesh, device, verbose=False, tag=f"l{layer_id}-init")
        torch_xla.sync(wait=True)
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

    @torch.compile(backend="tt")
    def run_embed(model, ids):
        h = model.embed(ids)
        return h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)

    @torch.compile(backend="tt")
    def run_block(block, h, sp, input_ids):
        return block(h, sp, input_ids)

    @torch.compile(backend="tt")
    def run_head(model, h):
        return model.head(h, model.hc_head_fn, model.hc_head_scale,
                          model.hc_head_base, model.norm)

    print("\n[infer] starting layer-streaming v4 ...", flush=True)
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))

    generated: List[List[int]] = [[] for _ in range(BATCH_SIZE)]
    next_input_ids = prompt_ids_tt
    sp_value = PROMPT_LEN

    for step in range(MAX_NEW_TOKENS):
        t_step = time.time()
        sp_tt = torch.tensor(sp_value if step == 0 else (PROMPT_LEN + step - 1),
                              dtype=torch.long).to(device)

        h = run_embed(skeleton, next_input_ids)
        torch_xla.sync(wait=True)
        h = _round_trip(h, device)
        xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
        _log(f"s{step} post-embed")

        for layer_id in range(NUM_LAYERS):
            t_layer = time.time()
            t_load_start = time.time()
            temp = _build_temp_layer(layer_id, args, mesh_shape)
            n = _ship_temp_into_skeleton(temp.layers[layer_id],
                                         skeleton.layers[layer_id],
                                         mesh, device)
            del temp
            gc.collect()
            t_load = time.time() - t_load_start

            t_exec_start = time.time()
            h_out = run_block(skeleton.layers[layer_id], h, sp_tt, next_input_ids)
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
            t_exec = time.time() - t_exec_start

            t_rt_start = time.time()
            h_next = _round_trip(h_out, device)
            xs.mark_sharding(h_next, mesh, ("_axis_0", None, None, None))
            t_rt = time.time() - t_rt_start

            del h, h_out
            gc.collect()
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
            gc.collect()

            h = h_next
            cr = args.compress_ratios[layer_id]
            t_total = time.time() - t_layer
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss / 1e9
            sys_used = psutil.virtual_memory().used / 1e9
            print(
                f"[infer s{step} l{layer_id:2d} cr={cr:3d}] "
                f"total={t_total:.2f}s load={t_load:.2f}s exec={t_exec:.2f}s rt={t_rt:.2f}s n={n} "
                f"rss={rss:.2f} sys={sys_used:.2f} GB",
                flush=True,
            )

        logits = run_head(skeleton, h)
        torch_xla.sync(wait=True)
        next_ids = logits.detach().to("cpu").argmax(dim=-1)
        del logits
        gc.collect()
        for i in range(BATCH_SIZE):
            generated[i].append(int(next_ids[i].item()))
        next_input_ids = next_ids.unsqueeze(1).to(device)
        xs.mark_sharding(next_input_ids, mesh, ("_axis_0", None))
        del h
        gc.collect()
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        print(f"[infer s{step}] total {time.time()-t_step:.2f}s ids[:4]={next_ids[:4].tolist()}", flush=True)

    print("\n[done] decoded:", flush=True)
    for i, (prompt, ids) in enumerate(zip(PROMPTS[:4], generated[:4])):
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  [{i}] {prompt!r} -> {cont!r}", flush=True)


if __name__ == "__main__":
    main()
