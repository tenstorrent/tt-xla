# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming-inference end-to-end test for DeepSeek-V4-Flash on Tenstorrent.

The full 43-layer MoE model is too large to hold in host RAM all at once, so
we stream it: build a weight-less skeleton, then for each transformer block
load its weights from HuggingFace, ship them (sharded) to device, and release
the host copy before moving to the next block. Peak host RAM stays bounded to
roughly one block's worth of staging.

The flow, top to bottom:

    build skeleton (no weights)
    ship top-level params (embed / norm / head)
    pre-allocate persistent KV buffers on device
    for block in model.layers:
        load HF weights → swap MoE → ship sharded → dummy-flush
    re-zero KV state
    model = torch.compile(model, backend="tt")
    prefill + decode loop

This is a run-to-completion smoke test: it generates tokens and prints the
decoded continuations, but does not check PCC against a CPU reference.

    pytest -svv tests/torch/models/deepseek_v4/test_deepseek_v4_e2e_streaming.py
"""
from __future__ import annotations

import gc
import logging
import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.sparse_mlp import enable_sparse_mlp
from ttxla_tools.logging import logger

from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)

from . import weight_loader

# ---- run configuration (intentionally fixed for this example) ----
MODEL_NAME = "deepseek-ai/DeepSeek-V4-Flash"
BATCH_SIZE = 8
MAX_NEW_TOKENS = 16
PROMPT_LEN = 128

# First BATCH_SIZE of these are tokenized and run.
PROMPTS = [
    "How are you today?",
    "What is the capital of France?",
    "Explain machine learning briefly.",
    "Who painted the Mona Lisa?",
    "What is two plus two?",
    "Tell me a fun fact about space.",
    "What is photosynthesis?",
    "How does a transformer model work?",
]


# ---------------------------------------------------------------------------
# Host-memory diagnostics
# ---------------------------------------------------------------------------
def _log_mem(tag: str) -> None:
    """Log host RSS so the bounded per-layer footprint is visible. Big weight
    tensors are mmap-backed, so freeing them (del + gc) returns the pages to
    the OS and RSS drops without an explicit malloc_trim."""
    import os

    import psutil

    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    sys_used = psutil.virtual_memory().used / 1e9
    logger.info(f"[mem {tag:24s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")


# ---------------------------------------------------------------------------
# Mesh + sharded upload helpers
# ---------------------------------------------------------------------------
def _make_mesh() -> Tuple[Mesh, Tuple[int, int]]:
    """2D device mesh: axis 0 = batch (data) parallel, axis 1 = model/tensor."""
    n = xr.global_runtime_device_count()
    if n == 32:
        mesh_shape = (4, 8)
    elif n == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, n)
    logger.info(f"[mesh] num_devices={n} mesh_shape={mesh_shape}")
    return Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1")), mesh_shape


def _upload(cpu_tensor: torch.Tensor, mesh, partition_spec, device) -> torch.Tensor:
    """Move a CPU tensor to the XLA device and annotate its shard spec.
    `partition_spec=None` leaves it replicated across all devices."""
    xla_t = cpu_tensor.to(device)
    if partition_spec is None:
        return xla_t
    if len(partition_spec) != cpu_tensor.dim():
        raise ValueError(
            f"partition_spec {partition_spec!r} rank != tensor shape "
            f"{tuple(cpu_tensor.shape)}"
        )
    xs.mark_sharding(xla_t, mesh, partition_spec)
    return xla_t


def _ship_module(module: nn.Module, spec_by_id: Dict[int, Tuple], mesh, device) -> None:
    """Replace every Parameter and Buffer in `module` with a device-resident,
    sharded copy. Drops the source CPU tensors so the caller can gc them.
    Tensors absent from `spec_by_id` upload replicated."""
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is None or p.device.type != "cpu":
                continue
            xla_t = _upload(p.data.detach(), mesh, spec_by_id.get(id(p)), device)
            sub._parameters[name] = nn.Parameter(xla_t, requires_grad=False)
        for name, b in list(sub._buffers.items()):
            if b is None or b.device.type != "cpu":
                continue
            xla_t = _upload(b.detach(), mesh, spec_by_id.get(id(b)), device)
            sub._buffers[name] = xla_t


# ---------------------------------------------------------------------------
# Buffer plumbing (persistent KV state lives on device across the run)
# ---------------------------------------------------------------------------
def _buffer_paths(block) -> List[Tuple[nn.Module, str, str]]:
    """(submodule, buffer_name, dotted_path) for every registered buffer."""
    out = []
    for sub_path, sub in block.named_modules():
        for name, buf in list(sub._buffers.items()):
            if buf is None:
                continue
            full = f"{sub_path}.{name}" if sub_path else name
            out.append((sub, name, full))
    return out


def _ship_persistent_buffers(block, mesh, device) -> Dict[str, torch.Tensor]:
    """Ship every CPU buffer in `block` to device (batch-sharded on axis 0 for
    rank>=3, else replicated). Returns {path: device_tensor} for re-splicing."""
    out: Dict[str, torch.Tensor] = {}
    for sub, name, full in _buffer_paths(block):
        b = sub._buffers[name]
        if b is None or b.device.type != "cpu":
            if b is not None:
                out[full] = b
            continue
        spec = (
            ("_axis_0",) + (None,) * (b.dim() - 1)
            if b.dim() >= 3
            else (None,) * b.dim()
        )
        xla_t = _upload(b.detach(), mesh, spec, device)
        sub._buffers[name] = xla_t
        out[full] = xla_t
    torch_xla.sync(wait=True)
    return out


def _splice_persistent_buffers(block, bufs: Dict[str, torch.Tensor]) -> None:
    """Point each block buffer at the persistent device tensor of the same path."""
    for sub, name, full in _buffer_paths(block):
        if full in bufs:
            sub._buffers[name] = bufs[full]


# DS V4 buffers the per-block dummy forward mutates and that are sensitive to
# their initial value. `score_state` is consumed by softmax(dim=1) and must
# start at -inf so unwritten slots get zero probability mass; the rest start
# at zero. Window-attn `kv_cache` is included because the dummy flush dirties
# it too.
_MUTABLE_BUFFERS = {"kv_cache", "kv_state", "score_state"}


def _mutable_init(name: str, shape, dtype) -> torch.Tensor:
    if name == "score_state":
        return torch.full(shape, float("-inf"), dtype=dtype)
    return torch.zeros(shape, dtype=dtype)


def _rezero_kv(block, bufs: Dict[str, torch.Tensor], mesh, device) -> None:
    """Re-ship clean init values for the mutable KV buffers a dummy flush
    dirtied, updating both the block and the persistent-buffer map."""
    for sub, name, full in _buffer_paths(block):
        if name not in _MUTABLE_BUFFERS:
            continue
        b = sub._buffers[name]
        if b is None:
            continue
        spec = (
            ("_axis_0",) + (None,) * (b.dim() - 1)
            if b.dim() >= 3
            else (None,) * b.dim()
        )
        cpu_init = _mutable_init(name, b.shape, b.dtype)
        xla_t = _upload(cpu_init, mesh, spec, device)
        sub._buffers[name] = xla_t
        bufs[full] = xla_t
        del cpu_init


# ---------------------------------------------------------------------------
# Model construction + per-model sharding specs (DeepSeek-V4-Flash)
# ---------------------------------------------------------------------------
def _build_skeleton() -> nn.Module:
    """Empty (weight-less) Transformer in bf16 sized for this run. Uses the
    full layer count from the model config."""
    args = weight_loader.load_config_args(MODEL_NAME, force_bf16=True)
    args.n_mtp_layers = 0
    args.max_batch_size = BATCH_SIZE

    # Compressor uses a fixed-size kv_cache (max_seq_len // ratio), so round
    # max_seq_len up to a multiple of every compress ratio that covers prompt
    # + decode.
    max_cr = max(args.compress_ratios) if args.compress_ratios else 0
    needed = PROMPT_LEN + MAX_NEW_TOKENS
    if max_cr > 0:
        args.max_seq_len = max(((needed + max_cr - 1) // max_cr) * max_cr, 2 * max_cr)
    else:
        args.max_seq_len = ((needed + 31) // 32) * 32

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)
    return model, args


def _top_level_spec(model) -> Dict[int, Tuple]:
    """embed / norm / head / hc_head_* are replicated (small enough that the
    all-gather/all-reduce of sharding them isn't worth it for Flash)."""
    spec = {
        model.embed.weight: (None, None),
        model.norm.weight: (None,),
        model.head.weight: (None, None),
        model.hc_head_fn: (None, None),
        model.hc_head_base: (None,),
        model.hc_head_scale: (None,),
    }
    return {id(t): ps for t, ps in spec.items()}


def _block_spec(block, mesh) -> Dict[int, Tuple]:
    """SPMD shard spec for one block: Megatron-pattern attention, compound-
    sharded routed experts, replicated shared experts."""
    compound = ("_axis_0", "_axis_1")
    specs: Dict[torch.Tensor, Tuple] = {}

    attn = block.attn
    specs[attn.wq_b.weight] = ("_axis_1", None)
    specs[attn.wo_a.weight] = ("_axis_1", None)
    specs[attn.wo_b.weight] = (None, "_axis_1")
    specs[attn.kv_cache] = ("_axis_0", None, None)
    if attn.compress_ratio:
        specs[attn.compressor.kv_cache] = ("_axis_0", None, None)
        specs[attn.compressor.kv_state] = ("_axis_0", None, None)
        specs[attn.compressor.score_state] = ("_axis_0", None, None)
        if getattr(attn, "indexer", None) is not None:
            specs[attn.indexer.wq_b.weight] = ("_axis_1", None)
            specs[attn.indexer.weights_proj.weight] = ("_axis_1", None)
            specs[attn.indexer.compressor.kv_cache] = ("_axis_0", None, None)
            specs[attn.indexer.compressor.kv_state] = ("_axis_0", None, None)
            specs[attn.indexer.compressor.score_state] = ("_axis_0", None, None)

    mlp = block.ffn.mlp
    specs[mlp.router.gate.weight] = (None, None)
    specs[mlp.experts.gate_proj] = (compound, None, None)
    specs[mlp.experts.up_proj] = (compound, None, None)
    specs[mlp.experts.down_proj] = (compound, None, None)
    shared = getattr(block.ffn, "shared_experts", None)
    if shared is not None:
        specs[shared.w1.weight] = (None, None)
        specs[shared.w2.weight] = (None, None)
        specs[shared.w3.weight] = (None, None)
    return {id(t): ps for t, ps in specs.items()}


def _strip_cpu_golden(block) -> None:
    """Drop the dense-MoE CPU references `enable_sparse_mlp` keeps for its own
    golden-eval fallback (~13 GB/block) — streaming only runs on device."""
    mlp = getattr(getattr(block, "ffn", None), "mlp", None)
    if mlp is None:
        return
    if hasattr(mlp, "_original_mlp"):
        object.__setattr__(mlp, "_original_mlp", None)
    experts = getattr(mlp, "experts", None)
    if experts is not None and "original_experts" in getattr(experts, "_modules", {}):
        del experts._modules["original_experts"]


def _load_block(block, layer_id: int) -> None:
    """Load layer `layer_id`'s HF weights into `block`, dequantizing fp4/fp8."""
    sd = weight_loader.load_block_state_dict(MODEL_NAME, layer_id)
    prefix = f"layers.{layer_id}."
    sd = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in sd.items()}
    result = block.load_state_dict(sd, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(
            f"layers.{layer_id}: unexpected keys {sorted(result.unexpected_keys)}"
        )
    del sd


def _tokenize(prompts: List[str]) -> torch.Tensor:
    """Wrap each prompt as <BOS><User>{prompt}<Assistant> and left-pad to
    PROMPT_LEN. Returns (BATCH_SIZE, PROMPT_LEN) input ids."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    bos_id, user_id = tok.bos_token_id, tok.convert_tokens_to_ids("<｜User｜>")
    asst_id = tok.convert_tokens_to_ids("<｜Assistant｜>")

    rows = []
    for i in range(BATCH_SIZE):
        body = tok(
            prompts[i % len(prompts)], return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        wrap = torch.tensor(
            [bos_id, user_id] + body.tolist() + [asst_id], dtype=torch.long
        )
        if wrap.shape[0] >= PROMPT_LEN:
            wrap = wrap[-PROMPT_LEN:]
        else:
            wrap = torch.cat(
                [
                    torch.full((PROMPT_LEN - wrap.shape[0],), pad_id, dtype=torch.long),
                    wrap,
                ]
            )
        rows.append(wrap)
    return torch.stack(rows, dim=0).contiguous(), tok


# ---------------------------------------------------------------------------
# Main streaming pipeline
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Route our INFO progress to stderr and quiet 3rd-party noise. The shared
    ttxla_tools logger defaults to WARNING, which would hide the per-layer
    progress lines."""
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")


@pytest.mark.nightly
@pytest.mark.bh_galaxy
@torch.inference_mode()
def test_streaming_dsv4_flash() -> None:
    _setup_logging()
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    # Per-layer flush emits one dynamo cache entry per unique (block, shape);
    # 43 layers needs more than the default 8.
    torch._dynamo.config.cache_size_limit = 1000
    # Keep const-eval inputs on device (not bounced to host) so the per-layer
    # host-RAM bound holds.
    torch_xla.set_custom_compile_options(
        {"enable_const_eval_inputs_to_system_memory": False}
    )

    mesh, mesh_shape = _make_mesh()
    device = torch_xla.device()
    if BATCH_SIZE % mesh_shape[0] != 0:
        raise ValueError(
            f"BATCH_SIZE ({BATCH_SIZE}) must divide the batch-axis device count "
            f"({mesh_shape[0]}); the batch is sharded on `_axis_0`."
        )

    t_run = time.time()

    # ---- skeleton (no weights) ----
    model, args = _build_skeleton()
    layers = list(model.layers)
    n_layers = len(layers)
    _log_mem("baseline")

    # freqs_cis is identical for layers sharing a compress ratio; alias them so
    # the per-layer dummy flush hits the compile cache instead of retracing.
    freqs_by_cr: Dict[int, torch.Tensor] = {}
    for blk in layers:
        cr = getattr(blk.attn, "compress_ratio", 0)
        if cr in freqs_by_cr:
            blk.attn.freqs_cis = freqs_by_cr[cr]
            if getattr(blk.attn, "compressor", None) is not None:
                blk.attn.compressor.freqs_cis = freqs_by_cr[cr]
        else:
            freqs_by_cr[cr] = blk.attn.freqs_cis

    # ---- ship top-level params (embed / norm / head / hc_head_*) ----
    t = time.time()
    embed_sd = weight_loader.load_embed_state_dict(MODEL_NAME)
    model.embed.load_state_dict(embed_sd, strict=False)
    del embed_sd
    model.load_state_dict(
        weight_loader.load_top_level_state_dict(MODEL_NAME), strict=False
    )
    gc.collect()
    top_spec = _top_level_spec(model)
    for sub in (model.embed, model.norm, model.head):
        _ship_module(sub, top_spec, mesh, device)
    for pname in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        p = model._parameters.get(pname)
        if p is not None and p.device.type == "cpu":
            xla_t = _upload(p.data.detach(), mesh, top_spec.get(id(p)), device)
            model._parameters[pname] = nn.Parameter(xla_t, requires_grad=False)
    torch_xla.sync(wait=True)
    gc.collect()
    logger.info(f"[step] top-level ship: {time.time() - t:.1f}s")
    _log_mem("post-top-level")

    # ---- pre-allocate persistent KV buffers on device (one set per layer) ----
    t = time.time()
    init_model, _ = _build_skeleton()
    init_layers = list(init_model.layers)
    persistent_bufs = [
        _ship_persistent_buffers(init_layers[i], mesh, device) for i in range(n_layers)
    ]
    del init_model, init_layers
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    logger.info(f"[step] kv-buffer preship: {time.time() - t:.1f}s")
    _log_mem("post-kv-preship")

    # ---- dummy block inputs (real prefill shape) for the per-layer flush ----
    h_cpu = torch.zeros(
        BATCH_SIZE, PROMPT_LEN, model.hc_mult, args.dim, dtype=torch.bfloat16
    )
    h_dummy = _upload(h_cpu, mesh, ("_axis_0", None, None, None), device)
    sp_dummy = torch.tensor(0, dtype=torch.long).to(device)
    ids_cpu = torch.zeros(BATCH_SIZE, PROMPT_LEN, dtype=torch.long)
    ids_dummy = _upload(ids_cpu, mesh, ("_axis_0", None), device)
    del h_cpu, ids_cpu
    torch_xla.sync(wait=True)

    @torch.compile(backend="tt")
    def run_block_flush(block, *blk_args):
        return block(*blk_args)

    # Per-layer stages, factored so the streaming loop below reads as just
    # `pre_flush → run_block_flush → post_flush`. They close over the run
    # state (mesh, device, args, persistent_bufs, dummy inputs) and stash
    # per-layer timing/hook on `_t` so the loop body stays three calls.
    _t: Dict[str, float] = {}

    def pre_flush(block, layer_id: int) -> None:
        """Everything before the dummy flush: load HF weights → swap dense MoE
        for the sparse/all-to-all version (dropping its CPU golden refs) →
        splice this layer's persistent KV buffers → ship the weights to device
        sharded → arm the output-shard hook for the upcoming flush."""
        cr = getattr(block.attn, "compress_ratio", None)
        logger.info(f"\n[stream] === layer {layer_id}/{n_layers - 1} (cr={cr}) ===")
        _t["start"] = time.time()

        _load_block(block, layer_id)
        gc.collect()
        enable_sparse_mlp(
            block, mesh=mesh_shape, cluster_axis=0, config=args, verbose=False
        )
        _strip_cpu_golden(block)
        gc.collect()
        _t["load"] = time.time() - _t["start"]

        _splice_persistent_buffers(block, persistent_bufs[layer_id])
        t_ship = time.time()
        _ship_module(block, _block_spec(block, mesh), mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        _t["ship"] = time.time() - t_ship

        block._flush_hook = block.register_forward_hook(
            sharding_constraint_hook(block, mesh, ("_axis_0", None, None, None))
        )

    def post_flush(block, layer_id: int) -> None:
        """Everything after the dummy flush: await it (the flush is what
        migrates the plugin's host staging buffers onto device, freeing host
        RAM for this layer), drop the hook, gc, and log per-layer progress."""
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        block._flush_hook.remove()
        del block._flush_hook
        gc.collect()

        total = time.time() - _t["start"]
        flush = total - _t["load"] - _t["ship"]
        logger.info(
            f"[stream l{layer_id}] total={total:.1f}s "
            f"load={_t['load']:.1f}s ship={_t['ship']:.1f}s flush={flush:.1f}s"
        )
        _log_mem(f"l{layer_id} post-flush")

    def reinit_kv(model) -> None:
        """The per-layer dummy flushes left every layer's mutable KV state
        dirty. Nothing reads it until prefill, so a single whole-model re-zero
        here (not per-layer) restores the clean init before the real run."""
        t = time.time()
        for li, block in enumerate(model.layers):
            _rezero_kv(block, persistent_bufs[li], mesh, device)
        torch_xla.sync(wait=True)
        logger.info(f"[step] KV re-zero: {time.time() - t:.1f}s")
        _log_mem("post-kv-rezero")

    # ====================== the streaming pipeline ======================
    t_loop = time.time()
    for layer_id in range(n_layers):
        block = layers[layer_id]
        pre_flush(block, layer_id)
        run_block_flush(block, h_dummy, sp_dummy, ids_dummy)
        post_flush(block, layer_id)
    logger.info(f"\n[step] per-layer loop: {time.time() - t_loop:.1f}s")

    reinit_kv(model)

    # Keep head logits replicated across devices.
    model.head.register_forward_hook(
        sharding_constraint_hook(model.head, mesh, (None, None))
    )

    # ---- whole-model compile, then prefill + decode ----
    logger.info("\n[stream] torch.compile(model) + prefill ...")
    compiled = torch.compile(model, backend="tt")

    prompts_used = [PROMPTS[i % len(PROMPTS)] for i in range(BATCH_SIZE)]
    ids_cpu, tok = _tokenize(prompts_used)
    prompt_ids = ids_cpu.to(device)
    xs.mark_sharding(prompt_ids, mesh, ("_axis_0", None))
    generated: List[List[int]] = [[] for _ in range(BATCH_SIZE)]

    # Prefill. Greedy argmax runs on device (traced into the graph) so only the
    # (bsz,) token ids cross to host, not the full (bsz, vocab) logits.
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    t = time.time()
    next_ids_tt = compiled(prompt_ids, sp).argmax(dim=-1)
    torch_xla.sync(wait=True)
    logger.info(f"[prefill] compile+exec {time.time() - t:.1f}s")
    next_ids = next_ids_tt.to("cpu")
    for i in range(BATCH_SIZE):
        generated[i].append(int(next_ids[i].item()))
    logger.info(f"[prefill] first ids[:8]={next_ids[:8].tolist()}")

    # Decode loop.
    prev = next_ids.unsqueeze(1)
    for step in range(MAX_NEW_TOKENS - 1):
        sp_val = PROMPT_LEN + step
        prev_tt = prev.to(device)
        xs.mark_sharding(prev_tt, mesh, ("_axis_0", None))
        sp = torch.tensor(sp_val, dtype=torch.long).to(device)
        t = time.time()
        next_ids_tt = compiled(prev_tt, sp).argmax(dim=-1)
        torch_xla.sync(wait=True)
        next_ids = next_ids_tt.to("cpu")
        for i in range(BATCH_SIZE):
            generated[i].append(int(next_ids[i].item()))
        kind = "compile+exec" if step == 0 else "exec"
        logger.info(
            f"[decode {step + 1}] sp={sp_val} {kind}={time.time() - t:.2f}s "
            f"ids[:8]={next_ids[:8].tolist()}"
        )
        prev = next_ids.unsqueeze(1)

    # ---- decode to text ----
    logger.info(f"\n[stream] done in {time.time() - t_run:.1f}s\n" + "=" * 72)
    eos_id = tok.eos_token_id
    for i, ids in enumerate(generated):
        trimmed = ids[: ids.index(eos_id)] if eos_id in ids else ids
        text = tok.decode(trimmed, skip_special_tokens=True)
        logger.info(f"[row {i:02d}] prompt={prompts_used[i]!r}")
        logger.info(f"          cont={text!r}")
    logger.info("=" * 72)
