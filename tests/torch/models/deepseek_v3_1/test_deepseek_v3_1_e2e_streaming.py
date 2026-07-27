# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming-inference end-to-end smoke test for DeepSeek-V3.1 (4 layers).

Runs the DeepSeek-V3.1 model (the one behind
``tests/benchmark/test_llms.py::test_deepseek_v3_1_tp_galaxy_4_layers``) with the
per-block *streaming* strategy first written for DeepSeek-V4-Flash
(``tests/torch/models/deepseek_v4/test_deepseek_v4_e2e_streaming.py``) and the
Kimi-K2 MLA streaming decode test: build a weight-less skeleton, then for each
transformer block load its weights from HuggingFace, ship them (sharded) to
device, force the lazy host->device transfer, and free the host copy before the
next block. Peak host RAM stays bounded to ~one block's worth of staging.

This is an INFRA smoke test, not an accuracy test. Only 4 layers are loaded
(``first_k_dense_replace=3`` -> layers 0-2 dense, layer 3 MoE), so the greedy
output is expected to be incoherent -- exactly like the benchmark's 4-layer run.
Validation is print + eyeball: we run prefill + greedy decode and print the
continuations; there is no PCC / accuracy assertion.

Unlike V4, DeepSeek-V3.1's KV cache is an *external* ``MLACache`` object (from
``tests/infra``) passed in as ``past_key_values``, so there is no per-block
persistent-buffer plumbing -- the cache is built once, sharded, and (because the
per-block dummy flushes dirty it) re-initialised with fresh device buffers before
the real run.

    pytest -svv tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_e2e_streaming.py
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
import sys
import time
import warnings
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

from . import weight_loader

# ---- run configuration ----
NUM_LAYERS = 12
BATCH_SIZE = 128
MAX_NEW_TOKENS = 16
# Block-float weight dtype for the resident (no-swap) weights. Streaming keeps all
# weights in device DRAM (enable_const_eval_inputs_to_system_memory=False), and
# bf16 weights are ~42 GB/chip for the full model -- far over the ~12 GB/chip DRAM.
# "bfp_bf8" ~halves the resident footprint, "bfp_bf4" ~quarters it (the full model
# is borderline resident at bf4); "" keeps bf16. Matches the Kimi streaming test.
WEIGHT_DTYPE = "bfp_bf8"  # "" | "bfp_bf8" | "bfp_bf4"
# Copied from tests/benchmark/benchmarks/llm_benchmark.py so the greedy output is
# directly comparable to the benchmark's perf-run print.
INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    """Route our INFO progress to stderr and quiet 3rd-party noise. The shared
    ttxla_tools/loguru logger defaults to WARNING, which would hide the per-block
    progress lines and the greedy continuations this test prints."""
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")


# ---------------------------------------------------------------------------
# Host-memory diagnostics
# ---------------------------------------------------------------------------
def _malloc_trim() -> None:
    """Return freed glibc arenas to the OS so RSS tracks live allocations.
    Without this, freed per-block CPU storage lingers in the process and the
    streaming host-RAM bound is impossible to observe."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _log_mem(tag: str) -> None:
    """Log host RSS so the bounded per-block footprint is visible."""
    _malloc_trim()
    try:
        import psutil

        rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
        sys_used = psutil.virtual_memory().used / 1e9
        logger.info(f"[mem {tag:24s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")
    except Exception:
        logger.info(f"[mem {tag:24s}] (psutil unavailable)")


# ---------------------------------------------------------------------------
# Mesh + sharded upload helpers
# ---------------------------------------------------------------------------
def _make_mesh(loader) -> Tuple[Mesh, Tuple[int, int]]:
    """Galaxy 2D device mesh: axis 0 = 'batch' (data parallel), axis 1 = 'model'
    (tensor parallel). Reuses the loader's own mesh config."""
    n = xr.global_runtime_device_count()
    mesh_shape, mesh_names = loader.get_mesh_config(n)
    logger.info(f"[mesh] num_devices={n} shape={mesh_shape} names={mesh_names}")
    return Mesh(np.arange(n), mesh_shape, mesh_names), mesh_shape


def _upload(cpu_tensor: torch.Tensor, mesh, partition_spec, device) -> torch.Tensor:
    """Move a CPU tensor to the XLA device (lazy) and annotate its shard spec.
    ``partition_spec=None`` leaves it replicated across all devices."""
    xla_t = cpu_tensor.to(device)
    if partition_spec is not None:
        xs.mark_sharding(xla_t, mesh, partition_spec)
    return xla_t


def _ship_module(module: nn.Module, spec_by_id: Dict[int, Tuple], mesh, device) -> None:
    """Replace every CPU Parameter and Buffer in ``module`` with a device-resident,
    (optionally) sharded copy, dropping the source CPU tensors. Tensors absent
    from ``spec_by_id`` upload replicated. The ``.to(device)`` is lazy -- it only
    executes when a computation consuming these tensors runs (the dummy flush)."""
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
# Rotary-cache materialization (the meta skeleton leaves them uninitialized)
# ---------------------------------------------------------------------------
def _materialize_rotary_caches(model, seq_len: int, dtype=torch.bfloat16) -> None:
    """Eagerly (re)compute every rotary embedding's inv_freq / cos / sin cache on
    CPU for the fixed rotary length ``seq_len`` (= max_cache_len).

    The skeleton is built on ``meta``, so each DeepseekV3*RotaryEmbedding's
    non-persistent ``inv_freq`` / ``cos_cached`` / ``sin_cached`` buffers are meta
    (they are ``persistent=False`` -> absent from the checkpoint -> never filled
    by ``load_state_dict``). Recomputing them here (YaRN rebuilds inv_freq from
    config scalars) and pinning ``max_seq_len_cached`` means the correct caches
    ship to device as ordinary held buffers, rather than being lazily -- and
    unreliably -- rebuilt inside a per-block flush graph."""
    count = 0
    for module in model.modules():
        if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "inv_freq"):
            module._set_cos_sin_cache(seq_len=seq_len, device="cpu", dtype=dtype)
            count += 1
    logger.info(f"[rotary] materialized {count} rotary cache(s) (seq_len={seq_len})")
    if count == 0:
        raise RuntimeError(
            "materialize_rotary_caches matched no rotary modules -- the rotary "
            "embedding API likely changed; update this helper."
        )


def _alias_rotary_across_layers(model) -> None:
    """Point every decoder layer's rotary embedding at layer 0's module.

    All layers share identical RoPE config, so their cos/sin/inv_freq tables are
    identical. Sharing a single module (identical buffer *objects*) lets the
    per-block dummy flush reuse the torch.compile cache instead of retracing for
    every layer: otherwise each block's distinct rotary buffers are const-folded
    as distinct graph constants and force a full ~recompile per MoE layer (the
    dominant streaming cost). Mirrors the V4 streaming test's freqs_cis aliasing.
    Call BEFORE materialization so only the shared module is built."""
    layers = model.model.layers
    shared = layers[0].self_attn.rotary_emb
    for layer in layers[1:]:
        layer.self_attn.rotary_emb = shared


# ---------------------------------------------------------------------------
# MLA cache helpers (external StaticCache-style cache, not registered buffers)
# ---------------------------------------------------------------------------
_KV_SPEC = ("batch", None, None, None)


def _init_mla_cache(config, batch_size: int, max_cache_len: int):
    """Create + eagerly allocate an MLA cache on CPU (bf16). Mirrors
    ``llm_utils.decode_utils.init_mla_cache`` so transfer/shard can run before any
    forward pass."""
    from infra import MLACache

    cache = MLACache(config=config, max_cache_len=max_cache_len)
    text_config = config.get_text_config(decoder=True)
    kv_lora_rank = text_config.kv_lora_rank
    qk_rope_head_dim = text_config.qk_rope_head_dim
    dummy_kv = torch.zeros((batch_size, 1, 1, kv_lora_rank), dtype=torch.bfloat16)
    dummy_pe = torch.zeros((batch_size, 1, 1, qk_rope_head_dim), dtype=torch.bfloat16)
    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)
    return cache


def _transfer_and_shard_cache(cache, mesh, device) -> None:
    """Move the MLA cache latents to device and batch-shard them."""
    for layer in cache.layers:
        layer.compressed_kv = layer.compressed_kv.to(device)
        layer.k_pe = layer.k_pe.to(device)
        layer.keys = layer.compressed_kv
        layer.values = layer.k_pe
        torch._dynamo.mark_static_address(layer.compressed_kv)
        torch._dynamo.mark_static_address(layer.k_pe)
        xs.mark_sharding(layer.compressed_kv, mesh, _KV_SPEC)
        xs.mark_sharding(layer.k_pe, mesh, _KV_SPEC)


def _reinit_cache_buffers(cache, mesh, device) -> None:
    """Replace the MLA cache latents with FRESH zeroed, sharded device buffers.

    The per-block dummy flushes both write garbage into the cache and (under
    host-input reclamation) free the host backing of the original shards. Ship
    brand-new zeroed device tensors so the real run consumes clean buffers whose
    host backing is still live."""
    for layer in cache.layers:
        fresh_kv = torch.zeros(
            tuple(layer.compressed_kv.shape), dtype=torch.bfloat16
        ).to(device)
        fresh_pe = torch.zeros(tuple(layer.k_pe.shape), dtype=torch.bfloat16).to(device)
        layer.compressed_kv = fresh_kv
        layer.k_pe = fresh_pe
        layer.keys = fresh_kv
        layer.values = fresh_pe
        torch._dynamo.mark_static_address(layer.compressed_kv)
        torch._dynamo.mark_static_address(layer.k_pe)
        xs.mark_sharding(layer.compressed_kv, mesh, _KV_SPEC)
        xs.mark_sharding(layer.k_pe, mesh, _KV_SPEC)


# ---------------------------------------------------------------------------
# Per-block / top-level SPMD shard specs (from _deepseek_v3_1_shard_spec_fn in
# tests/benchmark/test_llms.py). TP 8 (model) : DP 4 (batch) : EP 32.
# ---------------------------------------------------------------------------
def _block_shard_spec(layer) -> Dict[torch.Tensor, Tuple]:
    from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts

    specs: Dict[torch.Tensor, Tuple] = {}
    sa = layer.self_attn
    specs[sa.q_a_proj.weight] = (None, "model")
    specs[sa.q_b_proj.weight] = ("model", None)
    specs[sa.kv_a_proj_with_mqa.weight] = (None, "model")
    specs[sa.kv_b_proj.weight] = ("model", None)
    specs[sa.o_proj.weight] = (None, "model")
    specs[layer.input_layernorm.weight] = ("model",)
    specs[layer.post_attention_layernorm.weight] = ("model",)

    mlp = layer.mlp
    if isinstance(mlp, A2aSparseMLPWithSharedExperts):
        inner = mlp.mlp if hasattr(mlp, "mlp") else mlp
        specs[inner.router.gate.weight] = (None, "model")
        specs[inner.experts.gate_proj] = (("batch", "model"), None, None)
        specs[inner.experts.up_proj] = (("batch", "model"), None, None)
        specs[inner.experts.down_proj] = (("batch", "model"), None, None)
        for bias_name in ("gate_proj_bias", "up_proj_bias", "down_proj_bias"):
            b = getattr(inner.experts, bias_name, None)
            if b is not None:
                specs[b] = (("batch", "model"), None)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            specs[shared.gate_proj.weight] = (None, "model")
            specs[shared.up_proj.weight] = (None, "model")
            specs[shared.down_proj.weight] = ("model", None)
    else:
        specs[mlp.gate_proj.weight] = ("batch", "model")
        specs[mlp.up_proj.weight] = ("batch", "model")
        specs[mlp.down_proj.weight] = ("model", "batch")
    return specs


def _top_level_shard_spec(model) -> Dict[torch.Tensor, Tuple]:
    return {
        model.model.embed_tokens.weight: (None, "model"),
        model.model.norm.weight: ("model",),
        model.lm_head.weight: (None, "model"),
    }


def _strip_cpu_golden(block) -> None:
    """Drop the dense CPU-golden expert copies ``enable_sparse_mlp`` keeps for its
    CPU-fallback path (``A2aSparseMLP._original_mlp`` and
    ``StackedExperts.original_experts`` -- ~one block of experts *each*, referencing
    the same dense modules). Streaming runs on device only, so these are pure host
    overhead; without dropping them every MoE block leaks ~one block of experts and
    the per-block host-RAM bound is lost. No-op for dense layers. Mirrors the V4
    streaming test's ``_strip_cpu_golden``."""
    inner = getattr(getattr(block, "mlp", None), "mlp", None)  # A2aSparseMLP or None
    if inner is None:
        return
    if hasattr(inner, "_original_mlp"):
        object.__setattr__(inner, "_original_mlp", None)
    experts = getattr(inner, "experts", None)
    if experts is not None and "original_experts" in getattr(experts, "_modules", {}):
        del experts._modules["original_experts"]


# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------
def _build_skeleton(loader):
    """Weight-less DeepSeek-V3.1 (NUM_LAYERS layers) built on the meta device."""
    from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.modified_modeling_deepseek import (
        DeepseekV3ForCausalLM,
    )

    config = loader._load_config(num_layers=NUM_LAYERS)
    # Match the benchmark's setup_model_and_tokenizer config tweaks so MLACache
    # and the A2A sparse MLP behave identically.
    config._experts_implementation = "batched_mm"
    config.layer_types = ["full_attention"] * NUM_LAYERS
    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config).eval()
    return model, config


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.nightly
@pytest.mark.galaxy
@pytest.mark.tensor_parallel
@torch.inference_mode()
def test_streaming_deepseek_v3_1() -> None:
    from tt_torch.sharding import sharding_constraint_hook
    from tt_torch.sparse_mlp import enable_sparse_mlp

    from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    _setup_logging()
    do_flush = os.environ.get("NO_DUMMY_FLUSH", "0") != "1"

    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    torch.manual_seed(0)

    # One dynamo cache entry per unique (block, shape) during the per-block flush,
    # plus prefill + decode. The default 8 is far below what a streamed stack needs.
    torch._dynamo.config.cache_size_limit = 1000
    # Streaming knobs: match the benchmark's DeepSeek config (opt 0, no trace) and
    # keep const-eval inputs (the weights) in device DRAM instead of bouncing them
    # back to host, which would break the per-block host-RAM bound.
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": 0,
            "enable_trace": False,
            "enable_const_eval_inputs_to_system_memory": False,
            # Shrink the resident weights so more layers fit in device DRAM.
            "experimental_weight_dtype": WEIGHT_DTYPE,
        }
    )

    device = torch_xla.device()
    loader = ModelLoader(variant=ModelVariant.DEEPSEEK_V3_1_MODIFIED, num_layers=NUM_LAYERS)
    mesh, mesh_shape = _make_mesh(loader)
    if BATCH_SIZE % mesh_shape[0] != 0:
        raise ValueError(
            f"BATCH_SIZE ({BATCH_SIZE}) must divide the batch-axis device count "
            f"({mesh_shape[0]})."
        )

    t_run = time.time()

    # ---- tokenize the prompt (same prompt across the batch, like the benchmark) ----
    tokenizer = loader._load_tokenizer()
    enc = tokenizer(INPUT_PROMPT, return_tensors="pt")
    prompt_ids_1 = enc["input_ids"][0]
    prompt_len = int(prompt_ids_1.shape[0])
    prompt_ids_cpu = prompt_ids_1.unsqueeze(0).expand(BATCH_SIZE, -1).contiguous()
    # Fixed rotary / cache length covering prefill + all decode steps, tile-aligned.
    needed = prompt_len + MAX_NEW_TOKENS
    max_cache_len = ((needed + 31) // 32) * 32
    logger.info(
        f"[cfg] prompt_len={prompt_len} max_new_tokens={MAX_NEW_TOKENS} "
        f"max_cache_len={max_cache_len} batch={BATCH_SIZE} layers={NUM_LAYERS}"
    )

    # ---- 1. weight-less skeleton (meta) ----
    model, config = _build_skeleton(loader)
    layers = model.model.layers
    n_layers = len(layers)
    _log_mem("skeleton")

    # Share one rotary module across layers (so the per-block flush reuses the
    # compile cache instead of recompiling per layer), then materialize its caches
    # on CPU (the meta build left them uninitialized).
    _alias_rotary_across_layers(model)
    _materialize_rotary_caches(model, seq_len=max_cache_len, dtype=torch.bfloat16)

    # ---- 2. ship top-level params (embed / norm / lm_head) ----
    top_sd = weight_loader.load_top_level_state_dict(loader._BF16_WEIGHTS_REPO)
    _, unexpected = model.load_state_dict(top_sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"top-level load: unexpected keys {sorted(unexpected)[:8]}")
    del top_sd
    gc.collect()
    top_spec_by_id = {id(t): s for t, s in _top_level_shard_spec(model).items()}
    _ship_module(model.model.embed_tokens, top_spec_by_id, mesh, device)
    _ship_module(model.model.norm, top_spec_by_id, mesh, device)
    _ship_module(model.lm_head, top_spec_by_id, mesh, device)
    # All-gather the vocab-parallel lm_head logits so host-side argmax sees the
    # full vocabulary (mirrors the benchmark).
    model.lm_head.register_forward_hook(
        sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
    )
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()
    _log_mem("post-top-level")

    # ---- 3. allocate + ship the persistent MLA cache (on device before flush) ----
    cache = _init_mla_cache(config, BATCH_SIZE, max_cache_len)
    _transfer_and_shard_cache(cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log_mem("post-cache")

    # ---- 4. dummy decode inputs (q_len=1) for the per-block flush ----
    dummy_hidden = torch.zeros(
        BATCH_SIZE, 1, config.hidden_size, dtype=torch.bfloat16
    ).to(device)
    xs.mark_sharding(dummy_hidden, mesh, ("batch", None, None))
    cpu_cache_pos = torch.tensor([0], dtype=torch.long)
    dummy_mask = (
        cache.layers[0]
        .build_causal_mask(
            cache_position=cpu_cache_pos,
            batch_size=BATCH_SIZE,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        .to(device)
    )
    xs.mark_sharding(dummy_mask, mesh, ("batch", None, None, None))
    dummy_pos_ids = torch.tensor([[0]], dtype=torch.long).to(device)
    dummy_cache_pos = cpu_cache_pos.to(device)
    torch_xla.sync(wait=True)

    @torch.compile(backend="tt")
    def run_block_flush(block, hidden, mask, pos_ids, kv_cache, cache_pos):
        return block(
            hidden,
            attention_mask=mask,
            position_ids=pos_ids,
            past_key_value=kv_cache,
            use_cache=True,
            cache_position=cache_pos,
        )

    # ---- 5. per-block: load -> sparse-MLP -> ship -> dummy-flush -> free ----
    logger.info(f"[stream] streaming {n_layers} block(s) (dummy_flush={do_flush}) ...")
    t_loop = time.time()
    for layer_id in range(n_layers):
        t_blk = time.time()
        block = layers[layer_id]

        # 5a. load this block's HF weights into the meta skeleton (assign=True
        #     replaces the meta params with the real CPU tensors).
        block_sd = weight_loader.load_block_state_dict(
            loader._BF16_WEIGHTS_REPO, layer_id
        )
        _, unexpected = block.load_state_dict(block_sd, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(
                f"layer {layer_id} load: unexpected keys {sorted(unexpected)[:8]}"
            )
        del block_sd
        gc.collect()

        # 5b. swap dense MoE -> sparse all-to-all MLP (no-op for dense layers 0-2),
        #     then drop the dense CPU-golden expert copies it retains -- otherwise
        #     each MoE block leaks ~one block of experts (~22 GB) on the host.
        enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=config)
        _strip_cpu_golden(block)
        gc.collect()

        # 5c. ship the block sharded (lazy .to(device)).
        block_spec_by_id = {id(t): s for t, s in _block_shard_spec(block).items()}
        _ship_module(block, block_spec_by_id, mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()

        # 5d. dummy flush forces the lazy host->device transfer to execute so the
        #     plugin's host staging (and the CPU weight storage) can be released.
        if do_flush:
            run_block_flush(
                block, dummy_hidden, dummy_mask, dummy_pos_ids, cache, dummy_cache_pos
            )
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
        gc.collect()
        logger.info(
            f"[stream l{layer_id}] {time.time() - t_blk:.1f}s "
            f"(moe={type(block.mlp).__name__})"
        )
        _log_mem(f"l{layer_id} post-flush")
    logger.info(f"[step] per-block loop: {time.time() - t_loop:.1f}s")

    # ---- 6. re-init the MLA cache the flushes dirtied ----
    _reinit_cache_buffers(cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    _log_mem("post-stream")

    # ---- 7. whole-model compile + teacher-free greedy prefill + decode ----
    logger.info("[stream] torch.compile(model) + prefill ...")
    compiled = torch.compile(model, backend="tt")

    prompt_ids = prompt_ids_cpu.to(device)
    xs.mark_sharding(prompt_ids, mesh, ("batch", None))
    cache_position = torch.arange(0, prompt_len, dtype=torch.long).to(device)

    t = time.time()
    out = compiled(
        input_ids=prompt_ids,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
    )
    logits = out.logits.to("cpu").float()
    torch_xla.sync(wait=True)
    logger.info(f"[prefill] compile+exec {time.time() - t:.1f}s logits={tuple(logits.shape)}")

    next_ids = logits[:, -1].argmax(dim=-1)  # (batch,)
    n_rows = next_ids.shape[0]
    generated = [[int(next_ids[i])] for i in range(n_rows)]
    prev = next_ids.unsqueeze(1)

    for step in range(MAX_NEW_TOKENS - 1):
        input_ids = prev.to(device)
        xs.mark_sharding(input_ids, mesh, ("batch", None))
        cache_position = torch.tensor([prompt_len + step], dtype=torch.long).to(device)
        t = time.time()
        out = compiled(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = out.logits.to("cpu").float()
        torch_xla.sync(wait=True)
        next_ids = logits[:, -1].argmax(dim=-1)
        for i in range(n_rows):
            generated[i].append(int(next_ids[i]))
        prev = next_ids.unsqueeze(1)
        kind = "compile+exec" if step == 0 else "exec"
        logger.info(f"[decode {step}] sp={prompt_len + step} {kind}={time.time() - t:.2f}s")

    # ---- 8. print greedy continuations (incoherent at 4 layers -- expected) ----
    logger.info(f"[stream] done in {time.time() - t_run:.1f}s\n" + "=" * 72)
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    n_print = min(8, len(decoded))
    # NB: at 4 layers the logits are near-degenerate, so argmax is dominated by
    # numerical noise and rows need not agree even though the prompt is shared --
    # this is a print+eyeball smoke test, incoherent output is expected.
    lines = [f"[gen] prompt={INPUT_PROMPT!r} (first {n_print} of {len(decoded)} rows):"]
    for i in range(n_print):
        lines.append(f"  row {i:02d} -> {decoded[i]!r}")
    logger.info("\n".join(lines))
    logger.info("=" * 72)
