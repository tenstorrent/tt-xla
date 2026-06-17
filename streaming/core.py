# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic streaming inference runner.

`run_streaming(adapter, config)` orchestrates:

  1. Build CPU skeleton (via adapter).
  2. Ship top-level (embed/norm/head) to device.
  3. Pre-allocate persistent KV buffers.
  4. Per-layer load → ship → dummy-flush → KV re-init.
  5. Re-zero KV state.
  6. Whole-model `torch.compile` (whole_graph mode) OR per-layer compile
     (layer_eager mode).
  7. Tokenize prompts + run prefill + decode loop.

The adapter provides all model-specific knobs (block forward signature, KV
buffer names, MoE swap, weight-dtype overrides, tokenization, ...). See
`streaming.adapters.base.ModelAdapter`.
"""
from __future__ import annotations

import gc
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from tt_torch.sharding import sharding_constraint_hook, sharding_constraint_tensor
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as _mdo,
)
from tt_torch.weight_dtype import (
    apply_weight_carrier_overrides,
    apply_weight_dtype_overrides,
)
from ttxla_tools.logging import logger

from streaming._helpers import (
    _check_no_unexpected,
    _collect_buffer_paths,
    _log,
    _malloc_trim,
    _ship_persistent_buffers_raw,
    _ship_top_level,
    _splice_persistent_buffers,
)
from streaming.adapters.base import ModelAdapter
from streaming.config import StreamingConfig
from streaming.result import StreamingResult
from streaming.streaming_loader import _ship_module_handle_path, _upload_with_sharding


def _make_mesh():
    n = xr.global_runtime_device_count()
    if n == 32:
        mesh_shape = (4, 8)
    elif n == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, n)
    logger.info(f"[mesh] num_devices={n} mesh_shape={mesh_shape}")
    return Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1")), mesh_shape


def _reinit_mutable_kv_buffers(
    adapter,
    block,
    persistent_bufs_for_layer,
    mesh,
    device,
    mutable_names,
):
    """Re-ship fresh CPU buffers for each mutable KV-state buffer in
    `block`, using `adapter.mutable_buffer_init_value` so buffers with
    non-zero semantics (e.g. softmax-masked `score_state` → -inf) get the
    correct init. Updates `block._buffers[name]` AND
    `persistent_bufs_for_layer[full_path]` so all references point at the
    fresh copies."""
    for sub, name, full in _collect_buffer_paths(block):
        if name not in mutable_names:
            continue
        b = sub._buffers[name]
        if b is None:
            continue
        if b.dim() >= 3:
            partition_spec = ("_axis_0",) + (None,) * (b.dim() - 1)
        else:
            partition_spec = (None,) * b.dim()
        cpu_init = adapter.mutable_buffer_init_value(name, b.shape, b.dtype)
        xla_t = _upload_with_sharding(
            cpu_init,
            mesh,
            partition_spec,
            device,
        )
        sub._buffers[name] = xla_t
        persistent_bufs_for_layer[full] = xla_t
        del cpu_init
    torch_xla.sync(wait=True)


def _block_relative_overrides(weight_overrides: Dict[str, str]) -> Dict[str, str]:
    """Strip `layers.*.` prefix so overrides apply to a single Block."""
    relative: Dict[str, str] = {}
    layer_prefix = "layers.*."
    for pattern, dtype in weight_overrides.items():
        if pattern.startswith(layer_prefix):
            relative[pattern[len(layer_prefix) :]] = dtype
        elif pattern == "default":
            relative["default"] = dtype
    return relative


def run_streaming(
    adapter: ModelAdapter,
    config: StreamingConfig,
) -> StreamingResult:
    """Orchestrate streaming load + inference. See module docstring."""
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    # dynamo cache: per-layer flush emits one entry per unique (block, shape)
    # key + whole-model emits 2 (prefill + decode). Default 8 is way below
    # 43 layers; bump to 1000.
    torch._dynamo.config.cache_size_limit = 1000

    # Disable TTNNConstEvalInputsToSystemMemory so const-eval function
    # inputs stay in DEVICE storage instead of being moved to host. Must
    # be called after the TT plugin's ComputationClient is up (via
    # set_device_type) — calling at module-load triggers a duplicate
    # Initialize.
    custom_compile_options = {"enable_const_eval_inputs_to_system_memory": False}
    # Block-float expert weights are packed to their bfp tile dtype on host and
    # shipped straight to device, so the compiler retypes the weight args
    # directly instead of inserting a const-eval typecast/eviction. Gate on the
    # expert dtype being a block-float format.
    if config.expert_dtype and config.expert_dtype.lower() in (
        "bfp_bf4",
        "bfp_bf8",
    ):
        custom_compile_options["enable_host_packed_weights"] = True
    _early_ir = os.environ.get("STREAM_EARLY_IR_DUMP", "").strip()
    if _early_ir:
        os.makedirs(_early_ir, exist_ok=True)
        custom_compile_options["export_path"] = _early_ir
    torch_xla.set_custom_compile_options(custom_compile_options)

    mesh, mesh_shape = _make_mesh()
    device = torch_xla.device()
    bsz = config.batch_size

    # Pin the Hyper-Connection mix input's hidden `dim` replicated (batch stays
    # on _axis_0). With the hidden `dim` tensor-sharded on _axis_1, `hc_pre`'s
    # `flatten(2)` otherwise reshards via all_to_all -> O(N^2) point_to_point.
    # The constraint makes it an O(N) all_gather instead. No-op when the hidden
    # `dim` is already replicated (e.g. Flash).
    _mdo._HC_RESHARD_HOOK = lambda t: sharding_constraint_tensor(
        t, mesh, ("_axis_0",) + (None,) * (t.dim() - 1)
    )

    # Streaming shards the batch across mesh axis 0 (prompts, block KV
    # buffers, and `h` are all `("_axis_0", ...)`-sharded), so the batch must
    # divide the batch-axis device count. This is an implementation constraint
    # the caller needs to honor — assert it explicitly rather than failing
    # deep inside a sharding op (e.g. bsz=1 on a 4-wide batch axis).
    batch_devices = mesh_shape[0]
    if bsz % batch_devices != 0:
        raise ValueError(
            f"batch_size ({bsz}) must be divisible by the mesh batch-axis "
            f"device count ({batch_devices}); streaming shards the batch on "
            f"`_axis_0`."
        )

    weight_overrides = adapter.weight_dtype_overrides(
        config.expert_dtype,
        config.attn_dtype,
    )
    if weight_overrides:
        logger.info(
            f"[overrides] {adapter.name}: "
            f"expert={config.expert_dtype} attn={config.attn_dtype} → "
            f"{len(weight_overrides)} pattern(s)",
        )
    else:
        logger.info(
            f"[overrides] {adapter.name}: no dtype overrides "
            f"(expert={config.expert_dtype} attn={config.attn_dtype})",
        )

    timing: Dict[str, float] = {}
    t_run = time.time()

    # ---- skeleton ----
    t_section = time.time()
    model = adapter.build_skeleton(
        num_layers=config.num_layers,
        bsz=bsz,
        prompt_len=config.prompt_len,
        max_new_tokens=config.max_new_tokens,
    )
    layers = adapter.get_layers(model)
    n_layers = len(layers)
    _log("baseline")

    # Adapter-driven per-block tensor dedup (e.g. positional embeddings
    # shared across layers). Keeps the per-layer dummy flush hitting the
    # compile cache instead of retracing every layer.
    aliased = adapter.dedupe_per_block_tensors(layers)
    if aliased:
        logger.info(
            f"[stream] dedupe_per_block_tensors aliased {aliased}/{n_layers} layers"
        )

    # ---- top-level ship ----
    t_ship = time.time()
    _ship_top_level(
        model,
        mesh,
        device,
        top_level_shard_spec_fn=lambda m: adapter.top_level_shard_spec(m),
        load_embed_state_dict=adapter.load_embed_state_dict,
        load_top_level_state_dict=adapter.load_top_level_state_dict,
    )
    logger.info(
        f"[step] skeleton+top-level: skeleton={time.time() - t_section - (time.time() - t_ship):.1f}s "
        f"ship={time.time() - t_ship:.1f}s "
        f"total={time.time() - t_section:.1f}s",
    )

    # Top-level weight dtype override (e.g. head.weight → bfp4). Applied here,
    # after the top-level ship, so the const-eval typecast lands in the
    # whole-model (prefill) graph and frees the bf16 head before the first
    # prefill activation allocates.
    top_overrides = adapter.top_level_weight_dtype_overrides(config.head_dtype)
    if top_overrides:
        applied_top = apply_weight_dtype_overrides(model, top_overrides)
        for path, dtype_str in applied_top:
            logger.info(f"[wdtype] top-level {path} -> {dtype_str}")
        if not applied_top:
            logger.warning(
                f"[wdtype] top-level overrides matched nothing: {top_overrides}"
            )
    _log("post-top-level")

    # ---- pre-allocate persistent KV buffers ----
    logger.info("\n[stream] pre-allocating persistent KV buffers ...")
    t_section = time.time()
    init_skel = adapter.build_skeleton(
        num_layers=config.num_layers,
        bsz=bsz,
        prompt_len=config.prompt_len,
        max_new_tokens=config.max_new_tokens,
    )
    init_layers = adapter.get_layers(init_skel)
    persistent_bufs: List[Dict[str, torch.Tensor]] = []
    for layer_id in range(n_layers):
        bufs = _ship_persistent_buffers_raw(
            init_layers[layer_id],
            mesh,
            device,
        )
        persistent_bufs.append(bufs)
    del init_skel, init_layers
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    logger.info(f"[step] kv-buf-preship: total={time.time() - t_section:.1f}s")
    _log("post-init-buffers")

    # ---- dummy inputs for per-layer flush ----
    dummy_args = adapter.dummy_block_inputs(
        model,
        bsz,
        config.prompt_len,
        device,
        mesh,
    )
    logger.info(
        f"[stream] dummy forward shape: "
        f"bsz={bsz} seqlen={config.prompt_len} "
        f"(via adapter.dummy_block_inputs)",
    )
    torch_xla.sync(wait=True)

    @torch.compile(backend="tt")
    def run_block_flush(block, *args):
        return block(*args)

    mutable_names = adapter.mutable_buffer_names()

    # ---- per-layer ship + dummy-execute flush loop ----
    t_loop = time.time()
    for layer_id in range(n_layers):
        t_layer = time.time()
        cr = (
            getattr(layers[layer_id].attn, "compress_ratio", None)
            if hasattr(layers[layer_id], "attn")
            else None
        )
        cr_str = f" cr={cr}" if cr is not None else ""
        logger.info(
            f"\n[stream] === layer {layer_id} (loop {layer_id}/{n_layers - 1}){cr_str} ===",
        )
        block = layers[layer_id]

        # 1. Load HF weights.
        t_load = time.time()
        block_sd = adapter.load_block_state_dict(layer_id)
        _check_no_unexpected(
            block.load_state_dict(block_sd, strict=False),
            f"layers.{layer_id}",
        )
        del block_sd
        gc.collect()
        _log(f"l{layer_id} post-load")

        # 2. post_load_block hook (MoE swap, etc.)
        adapter.post_load_block(block, layer_id, mesh_shape)
        gc.collect()
        t_load = time.time() - t_load

        # 3. Splice persistent KV into block._buffers.
        _splice_persistent_buffers(block, persistent_bufs[layer_id])

        # 3b. Carrier mode (STREAM_EXPERT_CARRIER=1): pack block-float weights to
        # uint32 carriers + register carrier parametrizations BEFORE the ship, so
        # only the small carriers reach device (no bf16 weight ever lands there;
        # the PJRT relabels them to block-float tiles at input-prep). This REPLACES
        # the post-ship per-layer dtype override below. block_shard_spec then sees
        # the 2D carrier params (the adapter must give them a leading-dim spec).
        carrier_mode = os.environ.get("STREAM_EXPERT_CARRIER") == "1"
        import sys as _sys

        _sys.stderr.write(
            f"[CARRIER-DBG] layer {layer_id} carrier_mode={carrier_mode} "
            f"have_overrides={bool(weight_overrides)}\n"
        )
        _sys.stderr.flush()
        if carrier_mode and weight_overrides:
            applied_carrier = apply_weight_carrier_overrides(
                block, _block_relative_overrides(weight_overrides)
            )
            _sys.stderr.write(
                f"[CARRIER-DBG] layer {layer_id} applied {len(applied_carrier)} "
                f"carriers: {applied_carrier[:3]}\n"
            )
            _sys.stderr.flush()
            gc.collect()
            if layer_id == 0:
                logger.info(
                    f"[carrier] per-layer apply: {len(applied_carrier)} carriers",
                )
                for path, dtype_str in applied_carrier:
                    logger.info(f"[carrier]   {path} -> {dtype_str}")

        # 4. Ship CPU params to device.
        t_ship = time.time()
        block_specs = adapter.block_shard_spec(block, mesh)
        block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
        del block_specs
        _ship_module_handle_path(
            block,
            block_specs_by_id,
            mesh,
            device,
            verbose=False,
            tag=f"block-{layer_id}",
        )
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        t_ship = time.time() - t_ship

        # 4b. Per-layer weight dtype override (so const_eval typecast happens
        # at flush-compile time, before the next layer's weights add pressure).
        if weight_overrides and not carrier_mode:
            applied_block = apply_weight_dtype_overrides(
                block,
                _block_relative_overrides(weight_overrides),
            )
            if layer_id == 0:
                logger.info(
                    f"[wdtype] per-layer apply: {len(applied_block)} overrides per block",
                )
                for path, dtype_str in applied_block:
                    logger.info(f"[wdtype]   {path} -> {dtype_str}")
        # 5. Dummy execute (flush plugin-owned host staging buffers).
        t_flush = time.time()
        hook = block.register_forward_hook(
            sharding_constraint_hook(
                block,
                mesh,
                adapter.block_output_spec(block),
            )
        )
        try:
            _ = run_block_flush(block, *dummy_args)
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
        finally:
            hook.remove()

        # 6. Re-init mutable KV buffers.
        _reinit_mutable_kv_buffers(
            adapter,
            block,
            persistent_bufs[layer_id],
            mesh,
            device,
            mutable_names,
        )
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        t_flush = time.time() - t_flush

        gc.collect()
        _malloc_trim()

        t_total = time.time() - t_layer
        logger.info(
            f"[stream l{layer_id}{cr_str}] "
            f"total={t_total:.1f}s load={t_load:.1f}s "
            f"ship={t_ship:.1f}s flush={t_flush:.1f}s",
        )
        _log(f"l{layer_id} post-flush")
    timing["per_layer_loop_s"] = time.time() - t_loop

    # ---- all layers device-resident; final KV re-zero ----
    logger.info("\n[stream] all layers device-resident. Re-zero KV buffers ...")
    _log("post-all-layers")
    t_rezero = time.time()
    for li in range(n_layers):
        _reinit_mutable_kv_buffers(
            adapter,
            layers[li],
            persistent_bufs[li],
            mesh,
            device,
            mutable_names,
        )
    torch_xla.sync(wait=True)
    logger.info(f"[stream] re-zero done in {time.time() - t_rezero:.2f}s")
    _log("post-kv-rezero")

    # ---- head sharding hook (kept on the live model) ----
    if hasattr(model, "head"):
        head_hook_fn = sharding_constraint_hook(model.head, mesh, (None, None))
        model.head.register_forward_hook(head_hook_fn)

    # ---- inference: dispatch on mode ----
    if config.mode == "whole_graph":
        result = _run_inference_whole_graph(
            adapter,
            model,
            mesh,
            device,
            config,
        )
    elif config.mode == "layer_eager":
        result = _run_inference_layer_eager(
            adapter,
            model,
            mesh,
            device,
            config,
        )
    else:
        raise ValueError(f"Unknown mode: {config.mode!r}")

    timing["total_s"] = time.time() - t_run
    result.timing = timing
    return result


def _tokenize_and_ship_prompts(
    adapter: ModelAdapter,
    config: StreamingConfig,
    mesh,
    device,
) -> Tuple[torch.Tensor, List[str]]:
    """Tokenize first BATCH_SIZE prompts via adapter, mark sharding."""
    from streaming.adapters.deepseek_v4_flash import PROMPTS as _DEFAULT_PROMPTS

    # Adapters may expose their own canned prompt list; fall back to the
    # DS V4 Flash list so other adapters don't have to duplicate it.
    prompts_attr = getattr(adapter, "PROMPTS", None)
    if isinstance(prompts_attr, list):
        prompts = prompts_attr
    else:
        prompts = _DEFAULT_PROMPTS
    prompts_used = [prompts[i % len(prompts)] for i in range(config.batch_size)]

    prompt_ids = adapter.tokenize_prompts(
        prompts_used,
        config.batch_size,
        config.prompt_len,
    )
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    return prompt_ids_tt, prompts_used


def _run_inference_whole_graph(
    adapter: ModelAdapter,
    model,
    mesh,
    device,
    config: StreamingConfig,
) -> StreamingResult:
    """Whole-model `torch.compile` once, then prefill + decode."""
    logger.info("\n[stream] mode=whole_graph: torch.compile(model) ...")
    _log("pre-torch-compile")

    # Optional IR dump for the whole-model graph (prefill + decode). Set here
    # rather than alongside the const-eval option so the 60 per-layer flush
    # compiles don't each dump ~8 stage IRs. Writes <dir>/irs/*.mlir; the
    # final `ttnn_*.mlir` carries memory configs to locate large buffers.
    ir_dump_dir = os.environ.get("STREAM_IR_DUMP", "").strip()
    if ir_dump_dir:
        os.makedirs(ir_dump_dir, exist_ok=True)
        ir_opts = {
            "enable_const_eval_inputs_to_system_memory": False,
            "export_path": ir_dump_dir,
        }
        if config.expert_dtype and config.expert_dtype.lower() in (
            "bfp_bf4",
            "bfp_bf8",
        ):
            ir_opts["enable_host_packed_weights"] = True
        torch_xla.set_custom_compile_options(ir_opts)
        logger.info(f"[stream] IR dump enabled -> {ir_dump_dir}/irs/")

    t_section = time.time()
    compiled = torch.compile(model, backend="tt")
    logger.info(
        f"[step] torch.compile (lazy wrap, no XLA compile yet): "
        f"{time.time() - t_section:.2f}s",
    )
    _log("post-torch-compile")

    prompt_ids_tt, prompts_used = _tokenize_and_ship_prompts(
        adapter,
        config,
        mesh,
        device,
    )
    bsz = config.batch_size
    generated: List[List[int]] = [[] for _ in range(bsz)]

    sp_tt = torch.tensor(config.prompt_len, dtype=torch.long).to(device)
    _log("pre-prefill")
    t0 = time.time()
    prefill_logits = adapter.call_model(compiled, prompt_ids_tt, sp_tt)
    # Greedy argmax runs on device (traced into the same graph), so only the
    # (bsz,) token ids cross to host instead of the full (bsz, vocab) logits.
    next_ids_tt = prefill_logits.argmax(dim=-1)
    torch_xla.sync(wait=True)
    logger.info(f"[prefill] compile+exec {time.time() - t0:.1f}s")
    _log("post-prefill")

    next_ids = next_ids_tt.to("cpu")
    for i in range(bsz):
        generated[i].append(int(next_ids[i].item()))
    logger.info(f"[prefill] first ids[:8]={next_ids[:8].tolist()}")

    prev_token = next_ids.unsqueeze(1)
    for step in range(config.max_new_tokens - 1):
        sp_value = config.prompt_len + step
        prev_token_tt = prev_token.to(device)
        xs.mark_sharding(prev_token_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(sp_value, dtype=torch.long).to(device)
        t0 = time.time()
        decode_logits = adapter.call_model(compiled, prev_token_tt, sp_tt)
        next_ids_tt = decode_logits.argmax(dim=-1)
        torch_xla.sync(wait=True)
        elapsed = time.time() - t0
        next_ids = next_ids_tt.to("cpu")
        for i in range(bsz):
            generated[i].append(int(next_ids[i].item()))
        kind = "compile+exec" if step == 0 else "exec"
        logger.info(
            f"[decode {step + 1}] sp={sp_value} {kind}={elapsed:.2f}s "
            f"ids[:8]={next_ids[:8].tolist()}",
        )
        prev_token = next_ids.unsqueeze(1)

    return StreamingResult(
        generated_ids=generated,
        prompts_used=prompts_used,
    )


def _run_inference_layer_eager(
    adapter: ModelAdapter,
    model,
    mesh,
    device,
    config: StreamingConfig,
) -> StreamingResult:
    """Per-layer compile + sequential execute. Slower per-step than
    whole_graph, useful when whole-model compile is too memory-heavy or for
    per-layer profiling/debugging."""
    logger.info("\n[stream] mode=layer_eager: per-stage compile ...")
    _log("pre-stage-compile")
    layers = adapter.get_layers(model)

    @torch.compile(backend="tt")
    def pre_fn(model, ids, sp):
        return adapter.forward_pre_layers(model, ids, sp)

    @torch.compile(backend="tt")
    def layer_fn(layer, h, sp, ids):
        return adapter.forward_layer(layer, h, sp, ids)

    @torch.compile(backend="tt")
    def post_fn(model, h, ids):
        return adapter.forward_post_layers(model, h, ids)

    prompt_ids_tt, prompts_used = _tokenize_and_ship_prompts(
        adapter,
        config,
        mesh,
        device,
    )
    bsz = config.batch_size
    generated: List[List[int]] = [[] for _ in range(bsz)]

    def run_full(input_ids, start_pos):
        h = pre_fn(model, input_ids, start_pos)
        torch_xla.sync(wait=True)
        for layer in layers:
            h = layer_fn(layer, h, start_pos, input_ids)
            torch_xla.sync(wait=True)
        logits = post_fn(model, h, input_ids)
        torch_xla.sync(wait=True)
        return logits

    sp_tt = torch.tensor(config.prompt_len, dtype=torch.long).to(device)
    _log("pre-prefill")
    t0 = time.time()
    prefill_logits = run_full(prompt_ids_tt, sp_tt)
    logger.info(f"[prefill] compile+exec {time.time() - t0:.1f}s")
    _log("post-prefill")
    prefill_logits_cpu = prefill_logits.detach().to("cpu")

    next_ids = prefill_logits_cpu.argmax(dim=-1)
    for i in range(bsz):
        generated[i].append(int(next_ids[i].item()))
    logger.info(f"[prefill] first ids[:8]={next_ids[:8].tolist()}")

    prev_token = next_ids.unsqueeze(1)
    for step in range(config.max_new_tokens - 1):
        sp_value = config.prompt_len + step
        prev_token_tt = prev_token.to(device)
        xs.mark_sharding(prev_token_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(sp_value, dtype=torch.long).to(device)
        t0 = time.time()
        decode_logits = run_full(prev_token_tt, sp_tt)
        elapsed = time.time() - t0
        decode_logits_cpu = decode_logits.detach().to("cpu")
        next_ids = decode_logits_cpu.argmax(dim=-1)
        for i in range(bsz):
            generated[i].append(int(next_ids[i].item()))
        kind = "compile+exec" if step == 0 else "exec"
        logger.info(
            f"[decode {step + 1}] sp={sp_value} {kind}={elapsed:.2f}s "
            f"ids[:8]={next_ids[:8].tolist()}",
        )
        prev_token = next_ids.unsqueeze(1)

    return StreamingResult(
        generated_ids=generated,
        prompts_used=prompts_used,
    )


def print_decoded(
    adapter: ModelAdapter,
    result: StreamingResult,
) -> None:
    """Pretty-print result.generated_ids decoded back to text. Trims at
    first EOS per row."""
    logger.info("\n" + "=" * 72)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(adapter.tokenizer_repo_id)
        eos = tokenizer.eos_token_id
        for i, (prompt, ids) in enumerate(
            zip(result.prompts_used, result.generated_ids)
        ):
            cut = next((k for k, t in enumerate(ids) if t == eos), len(ids))
            ids_trim = ids[:cut]
            cont = tokenizer.decode(ids_trim, skip_special_tokens=True)
            logger.info(f"[row {i:02d}] prompt={prompt!r}")
            logger.info(f"         ids={ids_trim}")
            logger.info(f"         cont={cont!r}")
        logger.info("=" * 72)
    except Exception as e:
        logger.info(f"  (tokenizer decode failed: {e}; raw ids only)")
        for i, ids in enumerate(result.generated_ids):
            logger.info(f"  [row {i:02d}] {ids}")
