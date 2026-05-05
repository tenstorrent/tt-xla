"""[ARCHIVED] Layer-streaming inference v1 — template + weight swap.

Superseded by ../run_layer_stream.py. See archive/README.md for context.

Only 1 layer's weights live in the template at a time. Same compiled
binary serves all layers.

Architecture:
  - Top-level params (embed, head, norm) live on device permanently.
  - 'template' = model.layers[0] is the single Block instance compiled
    once via torch.compile.
  - For each layer_id, we hold a SEPARATE model instance (m_i) whose
    layers[layer_id] holds that layer's weights. (To start cheap, we
    pre-stream all N layers into N model instances, all device-resident.)
  - For each token step, we iterate layer_id 0..N-1:
      swap_weights(template, m_i.layers[layer_id])
      h = compiled_block(template, h, sp, input_ids)
  - swap_weights replaces template's _parameters with the per-layer
    weights. dynamo cache keys on sym_constants (shape/dtype), not
    handle id, so the binary is reused.

Usage:
    source venv/activate
    STREAM_NUM_LAYERS=2 STREAM_MAX_NEW_TOKENS=1 \
        python streaming/run_layer_stream.py
"""
from __future__ import annotations

import gc
import os
import time
from typing import List, Tuple

import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch import nn
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.weight_dtype import apply_weight_dtype_overrides
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
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "1"))
BATCH_SIZE = int(os.environ.get("STREAM_BATCH_SIZE", "32"))
NUM_LAYERS = int(os.environ.get("STREAM_NUM_LAYERS", "2"))

# When set, run *real* layer-streaming inference: only one layer's
# weights live on the device at a time. Each token step re-loads and
# re-ships every layer from HF in sequence. Slower but bounds device
# DRAM by 1 layer regardless of NUM_LAYERS.
RELEASE_PER_LAYER = bool(int(os.environ.get("STREAM_RELEASE_PER_LAYER", "0")))

EXPERT_DTYPE = "bfp_bf4"
ATTN_DTYPE = "bfp_bf8"

PROMPTS = [
    "How are you today?",
    "What is the capital of France?",
] + [f"Prompt {i}" for i in range(2, BATCH_SIZE)]
PROMPTS = PROMPTS[:BATCH_SIZE]


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
    print(f"[{tag:38s}] rss={rss:6.2f} sys_used={sys_used:6.2f} GB", flush=True)


def _build_model_skeleton(args):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)


def _ship_top_level(model, mesh, device, *, verbose=True):
    """Stream-load + ship top-level params (embed, head, norm, hc_*) to device."""
    # load_top_level_state_dict() omits 'embed.weight' (separate loader).
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
                             verbose=verbose, tag="top:embed")
    _ship_module_handle_path(model.norm, top_specs_by_id, mesh, device,
                             verbose=verbose, tag="top:norm")
    _ship_module_handle_path(model.head, top_specs_by_id, mesh, device,
                             verbose=verbose, tag="top:head")
    for pname in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        p = model._parameters.get(pname)
        if p is None or p.device.type != "cpu":
            continue
        partition_spec = top_specs_by_id.get(id(p))
        xla_t = _upload_with_sharding(p.data.detach(), mesh, partition_spec, device)
        model._parameters[pname] = nn.Parameter(xla_t, requires_grad=False)
    torch_xla.sync(wait=True)
    gc.collect()


def _direct_assign_block_weights(block, state_dict):
    """Populate block's parameters and buffers from state_dict via direct
    setattr. Bypasses load_state_dict to handle None slots (post-release).
    """
    for key, tensor in state_dict.items():
        parts = key.split(".")
        obj = block
        try:
            for part in parts[:-1]:
                obj = getattr(obj, part)
        except AttributeError:
            continue  # skip keys for submodules that don't exist
        last = parts[-1]
        # If it's a buffer slot, register as buffer; else as Parameter.
        if last in obj._buffers:
            obj._buffers[last] = tensor.contiguous()
        else:
            obj._parameters[last] = nn.Parameter(
                tensor.contiguous(), requires_grad=False,
            )


def _stream_load_layer_into(model, layer_id, mesh, mesh_shape, device, args, *, verbose=False):
    """Load HF weights for layer_id, sparse-rewrite, ship to model.layers[layer_id]."""
    block_sd = weight_loader.load_block_state_dict(layer_id)
    prefix = f"layers.{layer_id}."
    stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in block_sd.items()}
    # Direct setattr, bypassing load_state_dict (which struggles with
    # None slots after _release_block_weights).
    _direct_assign_block_weights(model.layers[layer_id], stripped)
    del block_sd, stripped
    gc.collect()
    enable_sparse_mlp(model.layers[layer_id], mesh=mesh_shape, cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(model.layers[layer_id])
    block = model.layers[layer_id]
    specs = _block_shard_spec(block, mesh)
    specs_by_id = {id(t): ps for t, ps in specs.items()}
    del specs
    _ship_module_handle_path(block, specs_by_id, mesh, device,
                             verbose=verbose, tag=f"layer-{layer_id}")
    torch_xla.sync(wait=True)
    gc.collect()


def _release_block_weights(block):
    """Drop all Parameter / Buffer refs on `block`.

    Replaces each slot with a meta tensor of same shape — keeps the
    Module structure valid (so subsequent direct_assign / load works)
    while releasing the XLA tensor backing.
    """
    if block is None:
        return
    import torch_xla.core.xla_model as xm
    for sub in block.modules():
        for name, p in list(sub._parameters.items()):
            if p is None:
                continue
            sub._parameters[name] = nn.Parameter(
                torch.empty(p.shape, dtype=p.dtype, device="meta"),
                requires_grad=False,
            )
        for name, b in list(sub._buffers.items()):
            if b is None:
                continue
            sub._buffers[name] = torch.empty(b.shape, dtype=b.dtype, device="meta")
    gc.collect()
    # Flush any pending xla ops so buffers can actually be destroyed.
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()


def _swap_weights(dst_block, src_block):
    """Replace dst_block's _parameters / _buffers / parametrizations with src's.
    Module identity of dst preserved → torch.compile cache hits.
    """
    dst_subs = list(dst_block.modules())
    src_subs = list(src_block.modules())
    assert len(dst_subs) == len(src_subs)
    for dst_sub, src_sub in zip(dst_subs, src_subs):
        for name in list(dst_sub._parameters.keys()):
            dst_sub._parameters[name] = src_sub._parameters.get(name)
        for name in list(dst_sub._buffers.keys()):
            dst_sub._buffers[name] = src_sub._buffers.get(name)
        if hasattr(dst_sub, "parametrizations") and hasattr(src_sub, "parametrizations"):
            dst_sub.parametrizations = src_sub.parametrizations


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
    # max_seq_len must be >= max compress_ratio (Compressor.kv_cache size
    # = max_seq_len // compress_ratio; would be 0 otherwise).
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    args.max_seq_len = max(
        ((PROMPT_LEN + MAX_NEW_TOKENS + 31) // 32) * 32,
        max_cr,
    )
    print(f"[args] layers={NUM_LAYERS} bsz={BATCH_SIZE} prompt={PROMPT_LEN} new={MAX_NEW_TOKENS}", flush=True)
    print(f"[args] max_seq_len={args.max_seq_len}", flush=True)

    _log("baseline")

    # ---- Build model skeleton + top-level on device ----
    print("\n[init] building primary model + top-level ...", flush=True)
    primary = _build_model_skeleton(args)
    _ship_top_level(primary, mesh, device, verbose=False)
    _log("post-top-level")

    # ---- Tokenize ----
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
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
    print(f"[tokenize] prompt_ids[0][-8:]={prompt_ids[0][-8:].tolist()}", flush=True)

    # ---- Layer model instances ----
    # We always pre-build N skeleton instances. The instance is reused
    # across token steps (so it's safe for dynamo cache). In release
    # mode the actual weights are loaded inside the inference loop,
    # then released after each layer's execute. In non-release mode all
    # layer weights are pre-loaded device-resident (the original path).
    layer_models: List[mdo.Transformer] = [primary]
    for layer_id in range(1, NUM_LAYERS):
        layer_models.append(_build_model_skeleton(args))

    if not RELEASE_PER_LAYER:
        print("\n[stream] pre-loading all layers (RELEASE_PER_LAYER=0) ...", flush=True)
        _stream_load_layer_into(primary, 0, mesh, mesh_shape, device, args)
        _log("post-layer-0")
        for layer_id in range(1, NUM_LAYERS):
            print(f"[stream] loading layer {layer_id} ...", flush=True)
            _stream_load_layer_into(layer_models[layer_id], layer_id, mesh,
                                    mesh_shape, device, args)
            _log(f"post-layer-{layer_id}")
    else:
        print(f"\n[stream] RELEASE_PER_LAYER=1 — layers loaded on demand", flush=True)

    # ---- Compile run_block once. Within a single variant (same
    # compress_ratio), dynamo + torch_xla cache should hit on subsequent
    # calls with different layer instances. Variant changes (new
    # compress_ratio) trigger a fresh compile.
    @torch.compile(backend="tt")
    def run_block(block, h, sp, input_ids):
        return block(h, sp, input_ids)

    @torch.compile(backend="tt")
    def run_embed(model, ids):
        # Mirrors Transformer.forward up to layer iteration.
        h = model.embed(ids)
        h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)
        return h

    @torch.compile(backend="tt")
    def run_head(model, h):
        return model.head(h, model.hc_head_fn, model.hc_head_scale,
                          model.hc_head_base, model.norm)

    # ---- Inference loop ----
    print("\n[infer] starting layer-streaming inference ...", flush=True)
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))

    generated: List[List[int]] = [[] for _ in range(BATCH_SIZE)]

    next_input_ids = prompt_ids_tt
    sp_value = PROMPT_LEN

    for step in range(MAX_NEW_TOKENS):
        t_step = time.time()
        sp_tt = torch.tensor(sp_value if step == 0 else (PROMPT_LEN + step - 1),
                              dtype=torch.long).to(device)

        # 1. embed
        h = run_embed(primary, next_input_ids)
        torch_xla.sync(wait=True)

        # 2. iterate layers
        prev_inst = None  # holds fresh model instance for release-mode
        for layer_id in range(NUM_LAYERS):
            t_layer = time.time()
            t_load = 0.0
            if RELEASE_PER_LAYER:
                # Drop previous fresh-instance: cascading Python ref drop
                # → all its XLA tensors freed → device DRAM + plugin host
                # owned tensor freed.
                if prev_inst is not None:
                    del prev_inst
                    gc.collect()
                    torch_xla.sync(wait=True)
                    import torch_xla.core.xla_model as xm
                    xm.wait_device_ops()
                    gc.collect()
                # Build fresh skeleton instance (cheap when bf16, mostly
                # uninitialized empty tensors).
                t_load_start = time.time()
                inst = _build_model_skeleton(args)
                _stream_load_layer_into(inst, layer_id, mesh, mesh_shape,
                                        device, args)
                block_i = inst.layers[layer_id]
                t_load = time.time() - t_load_start
            else:
                inst = layer_models[layer_id]
                block_i = inst.layers[layer_id]
            h = run_block(block_i, h, sp_tt, next_input_ids)
            torch_xla.sync(wait=True)
            cr = args.compress_ratios[layer_id]
            t_total = time.time() - t_layer
            t_exec = t_total - t_load
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss / 1e9
            sys_used = psutil.virtual_memory().used / 1e9
            print(
                f"[infer step={step} layer={layer_id:2d} cr={cr:3d}] "
                f"total={t_total:.2f}s load={t_load:.2f}s exec={t_exec:.2f}s "
                f"rss={rss:.2f} sys={sys_used:.2f} GB",
                flush=True,
            )
            prev_inst = inst if RELEASE_PER_LAYER else None

        # Release last instance at end of token step
        if RELEASE_PER_LAYER and prev_inst is not None:
            del prev_inst
            gc.collect()
            torch_xla.sync(wait=True)
            import torch_xla.core.xla_model as xm
            xm.wait_device_ops()
            gc.collect()

        # 3. head
        logits = run_head(primary, h)
        torch_xla.sync(wait=True)

        next_ids = logits.detach().to("cpu").argmax(dim=-1)
        for i in range(BATCH_SIZE):
            generated[i].append(int(next_ids[i].item()))
        next_input_ids = next_ids.unsqueeze(1).to(device)
        xs.mark_sharding(next_input_ids, mesh, ("_axis_0", None))
        if step == 0:
            sp_value = PROMPT_LEN
        print(f"[infer step={step}] total {time.time()-t_step:.2f}s ids[:4]={next_ids[:4].tolist()}", flush=True)

    print("\n[done] decoded:", flush=True)
    for i, (prompt, ids) in enumerate(zip(PROMPTS[:4], generated[:4])):
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"  [{i}] {prompt!r} -> {cont!r}", flush=True)


if __name__ == "__main__":
    main()
