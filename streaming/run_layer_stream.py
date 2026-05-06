"""Layer-streaming inference — fresh-instance + persistent kv_cache splicing.

Both host RAM and device DRAM are bounded to ~one layer's weights at a
time, while kv_cache state persists across token steps via externally-
managed device buffers that get spliced into each fresh CPU instance.

Architecture:
  - Persistent `kv_buffers[layer_id][buf_name]` dict of device tensors.
    Initialized once at startup (zeros). Persist across token steps.
    Updated in-place by run_block (kv_cache writes to the same device
    buffer).
  - Per-iter: build fresh model instance, splice kv_buffers[layer_id]
    into instance.layers[layer_id]._buffers, ship parameters, execute,
    capture updated buffer refs back into kv_buffers, then del the
    instance + round-trip h to release the layer's device buffers.

This is the canonical layer-streaming inference path. Earlier
prototypes (template+swap, persistent-skeleton-only, fresh-instance-
only, hybrid) live in `streaming/archive/` for reference.

Run:
    source venv/activate
    STREAM_NUM_LAYERS=2 python streaming/run_layer_stream.py

For per-layer PCC validation against a CPU eager reference:
    STREAM_INLINE_PCC=1 STREAM_NUM_LAYERS=2 python streaming/run_layer_stream.py
"""
from __future__ import annotations

import concurrent.futures
import gc
import os
import time
from typing import Dict, List

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
from tt_torch.sharding import sharding_constraint_hook
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
    _top_level_shard_spec, _upload_with_sharding,
)

# Each layer iter builds a fresh CPU block instance and del's it after
# shipping. dynamo's compiled-graph cache lines invalidate on weakref
# dealloc (one entry per unique block instance), so for N layers × M
# token steps we need cache_size_limit ≥ N*M. With NUM_LAYERS=43 and
# MAX_NEW_TOKENS up to 5 that's >200; bump well above the dynamo default
# of 8 to avoid the "recompile_limit hit → eager fallback" regression
# (https://github.com/tenstorrent/tt-xla/issues/4444). Same value as
# run_streaming.py.
torch._dynamo.config.cache_size_limit = 1000
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from streaming.pcc_utils import capture_or_compare, inline_pcc, REF_MODE, INLINE_PCC


PROMPT_LEN = int(os.environ.get("STREAM_PROMPT_LEN", "32"))
MAX_NEW_TOKENS = int(os.environ.get("STREAM_MAX_NEW_TOKENS", "1"))
BATCH_SIZE = int(os.environ.get("STREAM_BATCH_SIZE", "32"))
NUM_LAYERS = int(os.environ.get("STREAM_NUM_LAYERS", "4"))
# When set, run a single background worker that prefetches the next
# layer's CPU instance (`_build_layer_instance`) while the main thread
# is processing the current layer. Hides ~build_time/iter when
# build_time < (CPU eager + ship + exec). See OPEN_QUESTIONS.md #11b.
PIPELINE = os.environ.get("STREAM_PIPELINE", "1") == "1"
# Number of layers to prefetch ahead. depth=1 (default, safe) is
# single-lookahead. depth=N submits up to N layers ahead.
#
# **Caveats for depth>1**:
#   - Each in-flight layer adds ~13 GB CPU RAM (RSS).
#   - At full NUM_LAYERS=43, depth>1 has hit two issues:
#     (a) `pthread_create failed: Resource temporarily unavailable`
#         (per-user thread limit; check `ulimit -u`)
#     (b) `DefaultCPUAllocator: can't allocate memory` even with 400+
#         GB free — likely vm.max_map_count exhausted by the increased
#         number of live tensor mmap regions across in-flight layers.
#         Workaround: `sudo sysctl -w vm.max_map_count=262144`.
#   - On NUM_LAYERS=10, depth=2 saves ~7% wall time (372s → 346s);
#     depth=4 doesn't help further. So gain is small and the failure
#     modes can be hard to debug. Default kept at 1.
#   - Set OMP_NUM_THREADS / MKL_NUM_THREADS low (e.g., 4) when
#     experimenting with depth>1 to reduce thread pressure.
PIPELINE_DEPTH = max(1, int(os.environ.get("STREAM_PIPELINE_DEPTH", "1")))
# When set, wrap `run_block(...)` calls in cProfile for the first N
# layer iters (default N=3) and dump per-call stats after the run.
# Used to understand where t_trace's 9-21s actually goes (LTC IR
# build, dynamo dispatch, _run_cached_graph, etc.).
STREAM_PROFILE = int(os.environ.get("STREAM_PROFILE", "0"))
# When set, build one persistent device-resident "template" block per
# unique compress_ratio and reuse via in-place `param.data.copy_(...)`
# instead of fresh module instances each iter.
#
# **Tested 2026-05-01, parked**: dynamo cache DOES hit (the
# "Found an argument on non-XLA device" warning disappears entirely),
# but trace time is unchanged from the fresh-instance baseline:
#   baseline cr=0 l0/l1: 9.92s/8.86s    (cache miss, fresh inst)
#   template cr=0 l0/l1: 10.24s/8.93s   (cache hit)
# i.e., the per-call ~9s overhead lives in torch_xla's LTC-arg
# registration / extract_compiled_graph dispatch, NOT dynamo Python
# tracing. Plus 3 templates persistent on device → DRAM OOM at
# NUM_LAYERS≥10 (bsz=8 prompt=128). Net loss. See OPEN_QUESTIONS.md #11a.
#
# The flag is left in for future experiments.
TEMPLATE = os.environ.get("STREAM_TEMPLATE", "0") == "1"
# When set, the first step uses start_pos=0 (true prefill) instead of
# treating the prompt as decode-from-position-PROMPT_LEN. Matches
# test_transformer_prefill semantics. Default ON.
PREFILL_FIRST_STEP = os.environ.get("STREAM_PREFILL_FIRST", "1") == "1"
# When set, draw input_ids from realistic_inputs (same source as the
# test_transformer_prefill / test_transformer_decode tests). Default ON.
USE_REALISTIC_INPUTS = os.environ.get("STREAM_USE_REALISTIC", "1") == "1"

# When INLINE_PCC=1, halt the run if any layer's PCC drops below this
# threshold. Useful for finding the first divergence layer. Set to a
# very low value (e.g. -1) to never halt.
PCC_HALT_THRESHOLD = float(os.environ.get("STREAM_PCC_HALT_THRESHOLD", "0.98"))

# Optional disk cache for post-sparse_mlp CPU layer instances.
# When set, the first token's _build_layer_instance call writes a
# pickled module to disk; subsequent calls (decode tokens) reload it,
# skipping HF read + dequant + sparse_mlp (~25-30s saving / layer).
# Each cache file is ~13 GB BF16 — set this to a path with at least
# NUM_LAYERS × 13 GB free (e.g. /proj_sw/sshon/streaming_cache).
WEIGHT_CACHE_DIR = os.environ.get("STREAM_WEIGHT_CACHE_DIR", "")
if WEIGHT_CACHE_DIR:
    from pathlib import Path
    _wc_dir = Path(WEIGHT_CACHE_DIR)
    _wc_dir.mkdir(parents=True, exist_ok=True)
else:
    _wc_dir = None


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


def _ship_top_level(model, mesh, device, top_level_shard_spec_fn=None):
    # load_top_level_state_dict() returns head/norm/hc_head_* but NOT
    # embed.weight (which lives under its own loader function).
    embed_sd = weight_loader.load_embed_state_dict()
    model.embed.load_state_dict(embed_sd, strict=False)
    del embed_sd
    top_sd = weight_loader.load_top_level_state_dict()
    model.load_state_dict(top_sd, strict=False)
    del top_sd
    gc.collect()
    if top_level_shard_spec_fn is None:
        top_level_shard_spec_fn = _top_level_shard_spec
    top_specs = top_level_shard_spec_fn(model)
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


def _build_layer_instance(layer_id, args, mesh_shape, *, keep_cpu_golden=False):
    """Build a CPU model instance with layer_id's HF weights loaded
    and sparse-MLP rewritten.

    Optionally caches the post-sparse_mlp CPU block module to disk so
    subsequent calls (decode tokens) skip HF read + dequant + sparse_mlp.
    Cache file: {WEIGHT_CACHE_DIR}/l{N}_{golden|stripped}.pt.
    """
    cache_path = None
    if _wc_dir is not None:
        suffix = "golden" if keep_cpu_golden else "stripped"
        cache_path = _wc_dir / f"l{layer_id:02d}_{suffix}.pt"

    if cache_path is not None and cache_path.exists():
        # Cache hit — load post-sparse_mlp block straight into a fresh skeleton.
        m = _build_skeleton(args)
        block = torch.load(cache_path, map_location="cpu", weights_only=False)
        m.layers[layer_id] = block
        return m

    # Cache miss — normal build path.
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
    if not keep_cpu_golden:
        _strip_cpu_golden_refs(m.layers[layer_id])

    if cache_path is not None:
        # Save the post-sparse_mlp block module for future hits.
        torch.save(m.layers[layer_id], cache_path)

    return m


def _collect_buffer_paths(block):
    """Return list of (sub_module, buf_name, full_path) for every buffer
    in `block` so we can splice and capture by name."""
    out = []
    for sub_path, sub in block.named_modules():
        for name, buf in list(sub._buffers.items()):
            if buf is None:
                continue
            full = f"{sub_path}.{name}" if sub_path else name
            out.append((sub, name, full))
    return out


def _splice_persistent_buffers(block, persistent_bufs):
    """Replace block's _buffers with the persistent device tensors,
    keyed by full path. Persistent_bufs is a dict {full_path: xla_tensor}.
    """
    for sub, name, full in _collect_buffer_paths(block):
        if full in persistent_bufs:
            sub._buffers[name] = persistent_bufs[full]


def _capture_buffers(block):
    """After execute, capture the (possibly mutated) device tensors back
    into a dict by full path."""
    out: Dict[str, torch.Tensor] = {}
    for sub, name, full in _collect_buffer_paths(block):
        out[full] = sub._buffers[name]
    return out


def _ship_persistent_buffers_raw(block, mesh, device):
    """Ship a Block's persistent buffers without requiring the post-sparse_mlp
    module layout. The persistent buffers are all in `attn` (kv_cache,
    compressor.{kv_cache, kv_state, score_state}, optionally
    indexer.compressor.*) and shard on batch dim only — sparse_mlp doesn't
    touch them, so we can extract from a raw skeleton block.

    Used for the one-time persistent kv_cache buffer initialization, which
    only needs zero-init buffer shapes (no real weights needed).
    """
    out: Dict[str, torch.Tensor] = {}
    for sub, name, full in _collect_buffer_paths(block):
        b = sub._buffers[name]
        if b is None or b.device.type != "cpu":
            if b is not None:
                out[full] = b
            continue
        # All persistent KV/state buffers are 3D [batch, ...]; shard batch
        # along _axis_0. Anything < 3D (e.g. small scalars) replicates.
        if b.dim() >= 3:
            partition_spec = ("_axis_0",) + (None,) * (b.dim() - 1)
        else:
            partition_spec = (None,) * b.dim()
        xla_t = _upload_with_sharding(b.detach(), mesh, partition_spec, device)
        sub._buffers[name] = xla_t
        out[full] = xla_t
    torch_xla.sync(wait=True)
    return out


def _ship_buffers_only(block, mesh, device):
    """Walk block's _buffers; upload each CPU buffer via the handle
    path. After this, all buffers are device-resident XLA tensors.
    Returns dict {full_path: device_tensor}."""
    specs = _block_shard_spec(block, mesh)
    specs_by_id = {id(t): ps for t, ps in specs.items()}
    del specs
    out: Dict[str, torch.Tensor] = {}
    for sub, name, full in _collect_buffer_paths(block):
        b = sub._buffers[name]
        if b.device.type != "cpu":
            out[full] = b
            continue
        partition_spec = specs_by_id.get(id(b))
        xla_t = _upload_with_sharding(b.detach(), mesh, partition_spec, device)
        sub._buffers[name] = xla_t
        out[full] = xla_t
    torch_xla.sync(wait=True)
    return out


def _ship_parameters_only(block, mesh, device):
    """Walk block's _parameters; upload each CPU param. Returns count."""
    specs = _block_shard_spec(block, mesh)
    specs_by_id = {id(t): ps for t, ps in specs.items()}
    del specs
    n = 0
    for sub in block.modules():
        for name, p in list(sub._parameters.items()):
            if p is None or p.device.type != "cpu":
                continue
            partition_spec = specs_by_id.get(id(p))
            xla_t = _upload_with_sharding(p.data.detach(), mesh, partition_spec, device)
            sub._parameters[name] = nn.Parameter(xla_t, requires_grad=False)
            n += 1
    torch_xla.sync(wait=True)
    return n


def _round_trip(t, device):
    cpu = t.detach().to("cpu")
    return cpu.to(device)


# ---- Template-mode helpers ----

def _build_template(seed_layer_id, args, mesh, mesh_shape, device):
    """Build a persistent device-resident block instance to use as a
    cache-stable template for all layers sharing the same compress_ratio.

    Returns (inst, block). Caller persists `block` and keeps `inst` ref
    to avoid GC.
    """
    inst = _build_layer_instance(seed_layer_id, args, mesh_shape,
                                 keep_cpu_golden=False)
    block = inst.layers[seed_layer_id]
    _ship_buffers_only(block, mesh, device)
    _ship_parameters_only(block, mesh, device)
    # Force replicated output so the all_gather is fused into the block
    # forward graph (same trick as the non-template path).
    block.register_forward_hook(
        sharding_constraint_hook(block, mesh, (None, None, None, None))
    )
    torch_xla.sync(wait=True)
    return inst, block


def _inplace_copy_params_into(template_block, src_block):
    """In-place copy params of `src_block` into `template_block`.
    Preserves `template_block`'s param Python identity AND device
    storage location → dynamo cache hits.

    Triggers a sync after the bulk copy so XLA's temporary host→device
    upload buffers are released before the next iter (otherwise these
    accumulate per iter → device OOM).
    """
    src_params = dict(src_block.named_parameters())
    n = 0
    for name, dst_p in template_block.named_parameters():
        src_p = src_params.get(name)
        if src_p is None:
            continue
        dst_p.data.copy_(src_p.data)
        n += 1
    torch_xla.sync(wait=True)
    return n


def main():
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    n = xr.global_runtime_device_count()
    mesh_shape = (2, 4) if n == 8 else (4, 8) if n == 32 else (1, n)
    mesh = Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = BATCH_SIZE
    args.n_layers = NUM_LAYERS
    args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    # max_seq_len must be a multiple of max_cr AND ≥ 2*max_cr so the
    # Compressor.kv_cache (sized as max_seq_len // cr) has at least 2
    # slots — slot 0 for the prefill compress write, slot 1 for the
    # first decode write at sp=PROMPT_LEN where write_kv_idx=sp//cr=1.
    # Otherwise decode index_select trips OOB on cr=max_cr layers.
    needed = PROMPT_LEN + MAX_NEW_TOKENS
    if max_cr > 0:
        # Round up to next multiple of max_cr, with a floor of 2*max_cr.
        rounded = ((needed + max_cr - 1) // max_cr) * max_cr
        args.max_seq_len = max(rounded, 2 * max_cr)
    else:
        args.max_seq_len = ((needed + 31) // 32) * 32
    print(f"[args] layers={NUM_LAYERS} bsz={BATCH_SIZE} prompt={PROMPT_LEN} "
          f"new={MAX_NEW_TOKENS} max_seq_len={args.max_seq_len}", flush=True)

    _log("baseline")

    primary = _build_skeleton(args)
    _ship_top_level(primary, mesh, device)
    _log("post-top-level")

    # ---- Initialize persistent kv_cache buffers per layer ----
    # Buffers (kv_cache, kv_state, score_state, freqs_cis, ...) are zero-init
    # by Transformer.__init__ and don't depend on weight values, so a single
    # weight-less skeleton is sufficient — no HF read / dequant / sparse_mlp
    # per layer. Saves ~40s × NUM_LAYERS over the per-layer-instance variant.
    print("\n[init] persistent kv_cache buffers per layer (single-skeleton) ...", flush=True)
    init_skel = _build_skeleton(args)
    persistent_bufs: List[Dict[str, torch.Tensor]] = []
    for layer_id in range(NUM_LAYERS):
        bufs = _ship_persistent_buffers_raw(init_skel.layers[layer_id], mesh, device)
        persistent_bufs.append(bufs)
        _log(f"post-init-buffers-l{layer_id}")
    del init_skel
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log("post-init-buffers")

    if USE_REALISTIC_INPUTS:
        # Same source as test_transformer_prefill / test_transformer_decode:
        # cached deterministic tokenization sliced to (BATCH_SIZE, PROMPT_LEN).
        # n_hash_layers=3 -> activation distribution captured at the start of
        # the score-routed layers (matches what the tests use).
        from tests.torch.models.deepseek_v4 import realistic_inputs
        prompt_ids, _ = realistic_inputs.get_realistic_inputs(
            layer_id=args.n_hash_layers,
            batch_size=BATCH_SIZE,
            seq_len=PROMPT_LEN,
        )
        prompt_ids = prompt_ids.contiguous()
        print(f"[input] using realistic_inputs (layer={args.n_hash_layers}) "
              f"shape={tuple(prompt_ids.shape)}", flush=True)
    else:
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

    # ---- Template-mode setup ----
    # Build one device-resident block per unique compress_ratio. These
    # templates persist for the entire run; per-iter we copy_ the next
    # layer's weights into the template's existing parameter tensors,
    # preserving Python identity AND device storage location → dynamo
    # cache hits → trace-overhead per layer (9-21s) eliminated.
    templates: Dict[int, "nn.Module"] = {}
    template_insts: List["nn.Module"] = []
    if TEMPLATE:
        unique_crs = sorted(set(args.compress_ratios))
        print(f"\n[template] building {len(unique_crs)} templates (one per cr) ...",
              flush=True)
        for cr in unique_crs:
            seed_id = next(i for i, c in enumerate(args.compress_ratios) if c == cr)
            t0 = time.time()
            inst, block = _build_template(seed_id, args, mesh, mesh_shape, device)
            templates[cr] = block
            template_insts.append(inst)
            _log(f"template-cr{cr:>3d} seed=l{seed_id} built in {time.time()-t0:.1f}s")
        gc.collect()
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        _malloc_trim()

    print("\n[infer] starting layer-streaming ...", flush=True)
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))

    generated: List[List[int]] = [[] for _ in range(BATCH_SIZE)]
    next_input_ids = prompt_ids_tt
    # When PREFILL_FIRST_STEP: step 0 is true prefill (start_pos=0,
    # seq=PROMPT_LEN); subsequent steps are decode at PROMPT_LEN, PROMPT_LEN+1, ...
    # Otherwise: legacy semantics (start_pos=PROMPT_LEN throughout).

    for step in range(MAX_NEW_TOKENS):
        t_step = time.time()
        if PREFILL_FIRST_STEP:
            sp_value = 0 if step == 0 else (PROMPT_LEN + step - 1)
        else:
            sp_value = PROMPT_LEN if step == 0 else (PROMPT_LEN + step - 1)
        sp_tt = torch.tensor(sp_value, dtype=torch.long).to(device)

        h = run_embed(primary, next_input_ids)
        torch_xla.sync(wait=True)
        if REF_MODE != "none":
            capture_or_compare(step, -1, h, tag="embed")
        # Inline PCC for embed: compare CPU eager embed against device.
        if INLINE_PCC:
            ids_cpu = next_input_ids.detach().to("cpu")
            print(f"[debug] ids_cpu[0,:5]={ids_cpu[0,:5].tolist()}", flush=True)
            print(f"[debug] ids_cpu[1,:5]={ids_cpu[1,:5].tolist()}", flush=True)
            tmp_skel = _build_skeleton(args)
            embed_sd = weight_loader.load_embed_state_dict()
            tmp_skel.embed.load_state_dict(embed_sd, strict=False)
            del embed_sd
            top_sd = weight_loader.load_top_level_state_dict()
            tmp_skel.load_state_dict(top_sd, strict=False)
            del top_sd
            with torch.inference_mode():
                h_cpu_embed = tmp_skel.embed(ids_cpu)
                h_cpu_embed = h_cpu_embed.unsqueeze(2).repeat(1, 1, tmp_skel.hc_mult, 1)
            h_dev_unshard = h.detach().to("cpu").float()
            h_cpu_embed_f = h_cpu_embed.float()
            print(f"[debug] h_cpu_embed shape={tuple(h_cpu_embed_f.shape)} mean={h_cpu_embed_f.mean().item():.6f} std={h_cpu_embed_f.std().item():.6f}", flush=True)
            print(f"[debug] h_dev_unshard shape={tuple(h_dev_unshard.shape)} mean={h_dev_unshard.mean().item():.6f} std={h_dev_unshard.std().item():.6f}", flush=True)
            print(f"[debug] h_cpu[0,0,0,:8]={h_cpu_embed_f[0,0,0,:8].tolist()}", flush=True)
            print(f"[debug] h_dev[0,0,0,:8]={h_dev_unshard[0,0,0,:8].tolist()}", flush=True)
            print(f"[debug] h_cpu[1,0,0,:8]={h_cpu_embed_f[1,0,0,:8].tolist()}", flush=True)
            print(f"[debug] h_dev[1,0,0,:8]={h_dev_unshard[1,0,0,:8].tolist()}", flush=True)
            inline_pcc(step, -2, h_cpu_embed, h, tag="embed")
            del tmp_skel, h_cpu_embed, h_cpu_embed_f, h_dev_unshard
            gc.collect()
        h = _round_trip(h, device)
        xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
        _log(f"s{step} post-embed")

        # 1-step lookahead pipelining: prefetch layer N+1's CPU instance
        # in a background worker while we process layer N. See
        # OPEN_QUESTIONS.md #11b. Disabled with STREAM_PIPELINE=0.
        # Pre-populate prefetch queue: pending[i] is the future for layer i,
        # for i in [layer_cursor, layer_cursor+depth). At iter L we pop
        # pending[L] (wait), then submit layer L+depth (if any).
        from collections import deque as _deque
        executor = (
            concurrent.futures.ThreadPoolExecutor(max_workers=PIPELINE_DEPTH)
            if PIPELINE else None
        )
        pending: "_deque" = _deque()
        if PIPELINE:
            # Build layer 0 inline so the loop's first iter doesn't pay
            # the full cpu-build wait. Submit layers [1, depth) in parallel.
            inst0 = _build_layer_instance(
                0, args, mesh_shape, keep_cpu_golden=INLINE_PCC
            )
            done0 = concurrent.futures.Future()
            done0.set_result(inst0)
            pending.append(done0)
            for i in range(1, min(PIPELINE_DEPTH, NUM_LAYERS)):
                pending.append(executor.submit(
                    _build_layer_instance,
                    i, args, mesh_shape,
                    keep_cpu_golden=INLINE_PCC,
                ))
        for layer_id in range(NUM_LAYERS):
            t_layer = time.time()
            t_load_start = time.time()
            # In INLINE_PCC mode, keep CPU golden expert refs so the
            # eager CPU forward path inside sparse_mlp works.
            if PIPELINE:
                inst = pending.popleft().result()
                # Submit the next layer's build (keep `depth` in flight)
                next_id = layer_id + PIPELINE_DEPTH
                if next_id < NUM_LAYERS:
                    pending.append(executor.submit(
                        _build_layer_instance,
                        next_id, args, mesh_shape,
                        keep_cpu_golden=INLINE_PCC,
                    ))
            else:
                inst = _build_layer_instance(
                    layer_id, args, mesh_shape, keep_cpu_golden=INLINE_PCC,
                )
            block_i = inst.layers[layer_id]

            # ---- INLINE CPU REFERENCE (optional) ----
            # While inst still has CPU weights & buffers, run a CPU
            # eager forward to produce a reference output. Sync kv_cache
            # state from persistent device buffers first.
            h_cpu_out = None
            if INLINE_PCC:
                # Copy current device kv_cache state → temp CPU buffers
                for sub, name, full in _collect_buffer_paths(block_i):
                    dev_buf = persistent_bufs[layer_id].get(full)
                    if dev_buf is None:
                        continue
                    sub._buffers[name] = dev_buf.detach().to("cpu")
                # CPU input: pull current h to host
                h_cpu_in = h.detach().to("cpu")
                sp_cpu = sp_tt.detach().to("cpu")
                ids_cpu = next_input_ids.detach().to("cpu")
                with torch.inference_mode():
                    h_cpu_out = block_i(h_cpu_in, sp_cpu, ids_cpu)
                # NOW strip golden refs (256 original experts) so the
                # device ship below doesn't try to upload them.
                _strip_cpu_golden_refs(block_i)
                gc.collect()

            # ---- DEVICE PATH ----
            if TEMPLATE:
                cr_for_layer = args.compress_ratios[layer_id]
                template_block = templates[cr_for_layer]
                # In-place copy params (CPU) → template (device).
                # Preserves template's param identity → dynamo cache hits.
                n = _inplace_copy_params_into(template_block, block_i)
                # Splice this layer's persistent kv_cache buffers (device
                # tensors) into template by reference — forward mutates
                # them in place, no save-back needed afterwards.
                _splice_persistent_buffers(template_block, persistent_bufs[layer_id])
                # We have what we need on device; can drop the CPU inst.
                # (Template's forward_hook was registered at build time.)
                exec_block = template_block
            else:
                # Splice persistent kv_cache buffers (device tensors) into
                # this fresh CPU block so block uses the persistent state.
                _splice_persistent_buffers(block_i, persistent_bufs[layer_id])
                # Ship parameters only (buffers already device).
                n = _ship_parameters_only(block_i, mesh, device)
                # Force block output to be fully replicated so the all_gather
                # is fused into the block forward graph (otherwise reading
                # h_out triggers a separate all_gather graph compile).
                block_i.register_forward_hook(
                    sharding_constraint_hook(block_i, mesh, (None, None, None, None))
                )
                exec_block = block_i
            t_load = time.time() - t_load_start

            # Optionally profile the run_block call for the first
            # STREAM_PROFILE layer iters of step 0. Each layer dumps
            # its own pstats file so we can diff cold/warm calls.
            if STREAM_PROFILE > 0 and step == 0 and layer_id < STREAM_PROFILE:
                import cProfile, pstats, io
                _pr = cProfile.Profile()
                t_trace_start = time.time()
                _pr.enable()
                h_out = run_block(exec_block, h, sp_tt, next_input_ids)
                _pr.disable()
                t_trace = time.time() - t_trace_start
                # Dump top hot spots (cumulative time) to log.
                _buf = io.StringIO()
                pstats.Stats(_pr, stream=_buf).sort_stats(
                    "cumulative"
                ).print_stats(40)
                print(f"\n[profile l{layer_id} cr={args.compress_ratios[layer_id]} "
                      f"t_trace={t_trace:.2f}s] cumulative top 40:",
                      flush=True)
                print(_buf.getvalue(), flush=True)
            else:
                t_trace_start = time.time()
                h_out = run_block(exec_block, h, sp_tt, next_input_ids)
                t_trace = time.time() - t_trace_start
            t_sync_start = time.time()
            torch_xla.sync(wait=True)
            t_sync = time.time() - t_sync_start
            t_wait_start = time.time()
            xm.wait_device_ops()
            t_wait = time.time() - t_wait_start
            t_exec = t_trace + t_sync + t_wait

            # PCC hooks
            if INLINE_PCC and h_cpu_out is not None:
                pcc_val = inline_pcc(step, layer_id, h_cpu_out, h_out, tag="out")
                del h_cpu_out
                # NaN < x is always False, so halt explicitly on NaN/inf too.
                import math as _math
                bad = pcc_val is not None and (
                    _math.isnan(pcc_val) or _math.isinf(pcc_val)
                    or pcc_val < PCC_HALT_THRESHOLD
                )
                if bad:
                    print(
                        f"[halt] PCC {pcc_val} < threshold "
                        f"{PCC_HALT_THRESHOLD} at s{step} l{layer_id} "
                        f"cr={args.compress_ratios[layer_id]}. "
                        f"Stopping early for debug.",
                        flush=True,
                    )
                    return
            if REF_MODE != "none":
                capture_or_compare(step, layer_id, h_out, tag="out")

            if TEMPLATE:
                # Capture buffers from template back into persistent dict.
                # Forward may have written to spliced buffer refs (which
                # are persistent_bufs[layer_id]'s tensors), so this is a
                # no-op refresh for ref-equality but explicit for safety.
                persistent_bufs[layer_id] = _capture_buffers(exec_block)
            else:
                # Capture (possibly mutated) buffers back into persistent dict.
                persistent_bufs[layer_id] = _capture_buffers(block_i)

            # Block output is replicated (forced via forward_hook above);
            # re-annotate sharding for next layer's input (annotation only —
            # data is already replicated on every device, so no transfer).
            t_rt_start = time.time()
            h_next = h_out
            xs.mark_sharding(h_next, mesh, ("_axis_0", None, None, None))
            t_rt = time.time() - t_rt_start

            # Drop refs. Prefetched instances for layer L+1..L+depth-1
            # remain alive in `pending` until popped on their iter.
            del h, h_out, block_i, inst
            gc.collect()
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
            gc.collect()
            # Pipelining keeps PIPELINE_DEPTH+1 ~13 GB CPU instances
            # alive briefly per iter. With glibc's default arena allocator
            # that interleaved alloc/free pattern fragments the heap and
            # leaves freed pages stuck inside arenas. Force libc to return
            # non-arena-trapped pages to the OS each iter so RSS doesn't
            # creep upward indefinitely.
            _malloc_trim()

            h = h_next
            cr = args.compress_ratios[layer_id]
            t_total = time.time() - t_layer
            p = psutil.Process(os.getpid())
            rss = p.memory_info().rss / 1e9
            sys_used = psutil.virtual_memory().used / 1e9
            print(
                f"[infer s{step} l{layer_id:2d} cr={cr:3d}] "
                f"total={t_total:.2f}s load={t_load:.2f}s "
                f"exec={t_exec:.2f}s(trace={t_trace:.2f} sync={t_sync:.2f} wait={t_wait:.2f}) "
                f"rt={t_rt:.2f}s n={n} "
                f"rss={rss:.2f} sys={sys_used:.2f} GB",
                flush=True,
            )

        if executor is not None:
            # Drain any leftover prefetches (last few iters may not pop).
            for fut in pending:
                try:
                    fut.result(timeout=0)
                except Exception:
                    pass
            pending.clear()
            executor.shutdown(wait=True)
        gc.collect()

        logits = run_head(primary, h)
        torch_xla.sync(wait=True)
        if REF_MODE != "none":
            capture_or_compare(step, NUM_LAYERS, logits, tag="head")
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

    print("\n[done] generated tokens (first 4 rows):", flush=True)
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
        for i in range(min(4, BATCH_SIZE)):
            ids = generated[i]
            decoded_each = [tokenizer.decode([t]) for t in ids]
            joined = tokenizer.decode(ids)
            print(f"  [{i}] ids={ids}", flush=True)
            print(f"      per-token={decoded_each}", flush=True)
            print(f"      joined={joined!r}", flush=True)
    except Exception as e:
        print(f"  (tokenizer decode failed: {e}; raw ids only)", flush=True)
        for i in range(min(4, BATCH_SIZE)):
            print(f"  [{i}] {generated[i]}", flush=True)


if __name__ == "__main__":
    main()
