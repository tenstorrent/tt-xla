"""[ARCHIVED] Layer-streaming inference v2 — persistent skeleton + weight-only streaming.

Superseded by ../run_layer_stream.py. See archive/README.md for context.

Key design:
  - One persistent `skeleton` Transformer instance.
  - All kv_cache (and other) buffers are pre-shipped to device once at
    init and live there permanently. They accumulate state across token
    steps as expected.
  - For each token step, for each layer i:
      1. stream-load layer i's HF weights → ship to skeleton.layers[i]
         (replaces _parameters; old XLA tensors ref-drop and PJRT
         buffers destroyed).
      2. run_block(skeleton.layers[i], h, sp, ids) — execute.
      3. release weights: replace skeleton.layers[i]._parameters with
         meta-device tensors (zero memory; preserves shape metadata).

Memory at any given time on device:
  - top-level params (embed, head, norm, hc_head_*)
  - all 43 layers' kv_cache buffers (~few MB / layer)
  - exactly ONE layer's weights (the one currently executing)
  - intermediate activations

Run:
    source venv/activate
    STREAM_NUM_LAYERS=4 STREAM_MAX_NEW_TOKENS=2 \\
        python streaming/run_layer_stream_v2.py
"""
from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Tuple

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


def _log(tag: str) -> None:
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


def _materialize_layer_block_with_sparse(layer_id, args, mesh_shape):
    """Return a CPU Block instance with layer_id's weights loaded + sparse-MLP rewritten.

    Uses a temporary Transformer skeleton just to build the layer; we
    then return that one Block, dropping the rest of the skeleton.
    """
    block_sd = weight_loader.load_block_state_dict(layer_id)
    prefix = f"layers.{layer_id}."
    stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in block_sd.items()}
    del block_sd

    # Tiny skeleton just to host this single layer's Block.
    tiny_args = type(args)(**{**args.__dict__, "n_layers": 1, "compress_ratios": (args.compress_ratios[layer_id],)})
    tiny = _build_skeleton(tiny_args)
    tiny.layers[0].load_state_dict(stripped, strict=False)
    del stripped
    gc.collect()
    enable_sparse_mlp(tiny.layers[0], mesh=mesh_shape, cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(tiny.layers[0])
    block = tiny.layers[0]
    # Disconnect block from tiny so the rest of tiny can be GC'd
    del tiny
    gc.collect()
    return block


def _ship_block_weights(dst_block, src_block, mesh, device):
    """Ship src_block's CPU weights into dst_block as device-resident XLA
    tensors. Replaces dst_block._parameters; buffers (kv_cache!) are
    untouched on dst_block.

    Walks both dst and src in parallel. dst's _parameters slots get
    overwritten with new nn.Parameter wrapping XLA tensors uploaded via
    the handle path. Old XLA tensors in dst's slots get their refs
    dropped → backend buffer destroyed → device DRAM freed.
    """
    specs = _block_shard_spec(dst_block, mesh)
    specs_by_id_dst = {id(t): ps for t, ps in specs.items()}
    del specs
    dst_subs = list(dst_block.modules())
    src_subs = list(src_block.modules())
    assert len(dst_subs) == len(src_subs), \
        f"module mismatch dst={len(dst_subs)} src={len(src_subs)}"
    n_params = 0
    for dst_sub, src_sub in zip(dst_subs, src_subs):
        for name, src_p in list(src_sub._parameters.items()):
            if src_p is None:
                continue
            # Look up partition_spec from id of dst's CURRENT param (before replace).
            dst_p = dst_sub._parameters.get(name)
            partition_spec = (specs_by_id_dst.get(id(dst_p))
                              if dst_p is not None else None)
            cpu_data = src_p.data.detach()
            xla_t = _upload_with_sharding(cpu_data, mesh, partition_spec, device)
            dst_sub._parameters[name] = nn.Parameter(xla_t, requires_grad=False)
            n_params += 1
    torch_xla.sync(wait=True)
    gc.collect()
    return n_params


def _release_block_weights_to_meta(block):
    """Replace block's parameter slots with meta tensors (zero memory).
    Buffers are untouched (so kv_cache state is preserved).
    """
    for sub in block.modules():
        for name, p in list(sub._parameters.items()):
            if p is None:
                continue
            sub._parameters[name] = nn.Parameter(
                torch.empty(p.shape, dtype=p.dtype, device="meta"),
                requires_grad=False,
            )
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()


def main():
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    n_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4) if n_devices == 8 else (1, n_devices)
    mesh = Mesh(np.arange(n_devices), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = BATCH_SIZE
    args.n_layers = NUM_LAYERS
    args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    args.max_seq_len = max(((PROMPT_LEN + MAX_NEW_TOKENS + 31) // 32) * 32, max_cr)
    print(f"[args] layers={NUM_LAYERS} bsz={BATCH_SIZE} prompt={PROMPT_LEN} new={MAX_NEW_TOKENS} max_seq_len={args.max_seq_len}", flush=True)

    _log("baseline")

    # ---- Build persistent skeleton ----
    print("\n[init] building persistent skeleton ...", flush=True)
    skeleton = _build_skeleton(args)
    _log("post-skeleton")

    # ---- Ship top-level ----
    print("[init] shipping top-level ...", flush=True)
    _ship_top_level(skeleton, mesh, device)
    _log("post-top-level")

    # ---- For each layer, run sparse_mlp + ship one initial set of
    #      weights (so the layer's MoE structure is rewritten with
    #      stacked experts; kv_cache buffers populate). We use layer 0's
    #      weights as a placeholder for all layers — the actual
    #      per-layer weights get streamed in during inference.
    #      Buffers (kv_cache, freqs_cis) are populated here once. ----
    print("\n[init] initial per-layer setup (sparse_mlp + buffers) ...", flush=True)
    for layer_id in range(NUM_LAYERS):
        # Apply sparse_mlp on this layer using layer 0's HF weights.
        block_sd_0 = weight_loader.load_block_state_dict(0)
        prefix0 = "layers.0."
        stripped0 = {(k[len(prefix0):] if k.startswith(prefix0) else k): v
                     for k, v in block_sd_0.items()}
        skeleton.layers[layer_id].load_state_dict(stripped0, strict=False)
        del block_sd_0, stripped0
        gc.collect()
        enable_sparse_mlp(skeleton.layers[layer_id], mesh=mesh_shape,
                         cluster_axis=0, config=args, verbose=False)
        _strip_cpu_golden_refs(skeleton.layers[layer_id])
        gc.collect()
        # Ship buffers + initial weights to device.
        block_specs = _block_shard_spec(skeleton.layers[layer_id], mesh)
        block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
        del block_specs
        _ship_module_handle_path(skeleton.layers[layer_id], block_specs_by_id,
                                 mesh, device, verbose=False, tag=f"l{layer_id}-init")
        torch_xla.sync(wait=True)
        gc.collect()
        _log(f"post-init-layer-{layer_id}")

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

    # ---- Compile entry points ----
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

    # ---- Inference loop ----
    print("\n[infer] starting layer-streaming inference v2 ...", flush=True)
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

        for layer_id in range(NUM_LAYERS):
            t_layer = time.time()
            # Load this layer's weights into a temporary CPU Block (with
            # sparse-MLP rewrite applied), then ship to skeleton.layers[i].
            t_load_start = time.time()
            src_block = _materialize_layer_block_with_sparse(
                layer_id, args, mesh_shape,
            )
            n_params = _ship_block_weights(
                skeleton.layers[layer_id], src_block, mesh, device,
            )
            del src_block
            gc.collect()
            t_load = time.time() - t_load_start

            # Execute
            h = run_block(skeleton.layers[layer_id], h, sp_tt, next_input_ids)
            torch_xla.sync(wait=True)

            # Release THIS layer's weights to meta (kv_cache preserved).
            _release_block_weights_to_meta(skeleton.layers[layer_id])

            cr = args.compress_ratios[layer_id]
            t_total = time.time() - t_layer
            t_exec = t_total - t_load
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss / 1e9
            sys_used = psutil.virtual_memory().used / 1e9
            print(
                f"[infer step={step} layer={layer_id:2d} cr={cr:3d}] "
                f"total={t_total:.2f}s load={t_load:.2f}s exec={t_exec:.2f}s "
                f"params={n_params} rss={rss:.2f} sys={sys_used:.2f} GB",
                flush=True,
            )

        logits = run_head(skeleton, h)
        torch_xla.sync(wait=True)
        next_ids = logits.detach().to("cpu").argmax(dim=-1)
        for i in range(BATCH_SIZE):
            generated[i].append(int(next_ids[i].item()))
        next_input_ids = next_ids.unsqueeze(1).to(device)
        xs.mark_sharding(next_input_ids, mesh, ("_axis_0", None))
        print(f"[infer step={step}] total {time.time()-t_step:.2f}s ids[:4]={next_ids[:4].tolist()}", flush=True)

    print("\n[done] decoded:", flush=True)
    for i, (prompt, ids) in enumerate(zip(PROMPTS[:4], generated[:4])):
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  [{i}] {prompt!r} -> {cont!r}", flush=True)


if __name__ == "__main__":
    main()
