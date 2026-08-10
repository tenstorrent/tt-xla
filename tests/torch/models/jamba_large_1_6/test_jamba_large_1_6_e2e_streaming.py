# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming-inference end-to-end test for AI21-Jamba-Large-1.6 on Tenstorrent.

Jamba Large (~398B total / ~94B active) is a hybrid Mamba + Attention + MoE
model (``JambaForCausalLM``). The full stack is too large to hold in host RAM,
so we stream it: meta skeleton → per-layer HF load (fuse MoE) → CPU golden →
ship sharded → dummy flush → whole-model compile → prefill/decode PCC + gen.

MoE: on galaxy-bh use HF ``tt_dense`` with 2D expert-weight sharding (``tt_moe``
is known-broken on galaxy — tt-xla#3941). Dense MLP / attention use 1D TP on
``_axis_1``. Mamba mixer weights are replicated (small vs MoE experts).
CPU golden uses ``eager`` experts + ``use_mamba_kernels=False`` (slow path).

BFP8 weights + DRAM space-saving are required to fit on galaxy-bh (same pattern
as DeepSeek-V4 streaming, tt-xla#5822).

Requires HF access to the gated repo (``HF_TOKEN`` / ``huggingface-cli login``).

    pytest -svv tests/torch/models/jamba_large_1_6/test_jamba_large_1_6_e2e_streaming.py
"""

from __future__ import annotations

import copy
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
from transformers import AutoTokenizer, DynamicCache, JambaForCausalLM
from transformers.masking_utils import create_causal_mask
from transformers.models.jamba.modeling_jamba import (
    JambaAttentionDecoderLayer,
    JambaMambaDecoderLayer,
    JambaMLP,
    JambaSparseMoeBlock,
)
from tt_torch.moe_backend import (
    TT_DENSE_EXPERTS_BACKEND_NAME,
    TT_MOE_BACKEND_NAME,
    register_tt_moe_backend,
)
from tt_torch.sharding import sharding_constraint_hook
from ttxla_tools.logging import logger

from tests.benchmark.utils import compute_pcc

from . import weight_loader

# ---- run configuration ----
MODEL_NAME = weight_loader.MODEL_NAME
BATCH_SIZE = 8
MAX_NEW_TOKENS = 16
PROMPT_LEN = 128

# Same as runner YAML ``enable_weight_bfp8_conversion: true`` →
# compiler_config.experimental_weight_dtype = "bfp_bf8".
# Required on galaxy-bh: bf16 OOM; BFP8 fits (see tt-xla#5822).
ENABLE_WEIGHT_BFP8_CONVERSION = True
PREFER_TT_MOE = True

# Prefill drifts more than a single decode step under BFP8, so hold a looser bar.
PREFILL_PCC_BAR = 0.90
DECODE_PCC_BAR = 0.95

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


def _log_mem(tag: str) -> None:
    import os

    import psutil

    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    sys_used = psutil.virtual_memory().used / 1e9
    logger.info(f"[mem {tag:24s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")


def _make_mesh() -> Tuple[Mesh, Tuple[int, int]]:
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


def _select_experts_backend(num_experts: int, mesh_shape: Tuple[int, int]) -> str:
    batch_axis, model_axis = mesh_shape
    if batch_axis > 1 and model_axis > 1:
        return TT_DENSE_EXPERTS_BACKEND_NAME
    if PREFER_TT_MOE and model_axis > 1 and num_experts % model_axis == 0:
        return TT_MOE_BACKEND_NAME
    if PREFER_TT_MOE and batch_axis > 1 and num_experts % batch_axis == 0:
        return TT_MOE_BACKEND_NAME
    return TT_DENSE_EXPERTS_BACKEND_NAME


def _build_skeleton(experts_backend: str) -> Tuple[JambaForCausalLM, object]:
    register_tt_moe_backend(cluster_axis=1)
    config = weight_loader.load_config(MODEL_NAME)
    config._experts_implementation = experts_backend
    config.use_cache = True
    config.use_mamba_kernels = False

    with torch.device("meta"):
        model = JambaForCausalLM(config)
    model.eval()
    return model, config


def _materialize_and_load(
    module: nn.Module, state_dict: Dict[str, torch.Tensor]
) -> None:
    result = module.load_state_dict(state_dict, strict=False, assign=True)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys: {sorted(result.unexpected_keys)}")
    own = set(module.state_dict().keys())
    still_missing = [k for k in result.missing_keys if k in own]
    if still_missing:
        raise RuntimeError(f"missing keys: {still_missing[:16]}")


def _expert_weight_spec(experts_backend: str) -> Tuple[Tuple, Tuple]:
    if experts_backend == TT_MOE_BACKEND_NAME:
        return ("_axis_1", None, None), ("_axis_1", None, None)
    # tt_dense galaxy: gate_up [E, 2I, H], down [E, H, I]
    return ("_axis_1", "_axis_0", None), ("_axis_1", None, "_axis_0")


def _top_level_spec(model: JambaForCausalLM) -> Dict[int, Tuple]:
    return {
        id(model.model.embed_tokens.weight): (None, None),
        id(model.model.final_layernorm.weight): (None,),
        id(model.lm_head.weight): ("_axis_1", None),
    }


def _block_spec(layer: nn.Module, experts_backend: str) -> Dict[int, Tuple]:
    """Shard specs for one Jamba decoder layer (attention or mamba + FFN)."""
    specs: Dict[torch.Tensor, Tuple] = {}

    if isinstance(layer, JambaAttentionDecoderLayer):
        attn = layer.self_attn
        specs[attn.q_proj.weight] = ("_axis_1", None)
        specs[attn.k_proj.weight] = ("_axis_1", None)
        specs[attn.v_proj.weight] = ("_axis_1", None)
        specs[attn.o_proj.weight] = (None, "_axis_1")
    elif isinstance(layer, JambaMambaDecoderLayer):
        # Mamba footprint is small vs MoE experts — leave unreplicated-spec
        # (default full replicate) for bring-up.
        pass
    else:
        raise TypeError(f"unsupported layer type: {type(layer)}")

    specs[layer.input_layernorm.weight] = (None,)
    specs[layer.pre_ff_layernorm.weight] = (None,)

    ff = layer.feed_forward
    if isinstance(ff, JambaSparseMoeBlock):
        specs[ff.router.weight] = (None, None)
        gu_spec, down_spec = _expert_weight_spec(experts_backend)
        specs[ff.experts.gate_up_proj] = gu_spec
        specs[ff.experts.down_proj] = down_spec
    elif isinstance(ff, JambaMLP):
        specs[ff.gate_proj.weight] = ("_axis_1", None)
        specs[ff.up_proj.weight] = ("_axis_1", None)
        specs[ff.down_proj.weight] = (None, "_axis_1")
    else:
        raise TypeError(f"unsupported feed_forward type: {type(ff)}")

    return {id(t): ps for t, ps in specs.items()}


def _load_block(layer: nn.Module, layer_id: int, config) -> None:
    sd = weight_loader.load_block_state_dict(
        MODEL_NAME, layer_id, num_experts=config.num_experts, config=config
    )
    _materialize_and_load(layer, sd)
    del sd


def _tokenize(prompts: List[str]):
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    texts = []
    for i in range(BATCH_SIZE):
        conversation = [{"role": "user", "content": prompts[i % len(prompts)]}]
        if tok.chat_template is not None:
            texts.append(
                tok.apply_chat_template(
                    conversation, tokenize=False, add_generation_prompt=True
                )
            )
        else:
            texts.append(prompts[i % len(prompts)])

    encoded = tok(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=PROMPT_LEN,
        add_special_tokens=False,
    )
    return encoded.input_ids.contiguous(), encoded.attention_mask.contiguous(), tok


def _causal_mask(config, inputs_embeds, attention_mask, past_key_values, position_ids):
    return create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )


def _setup_logging() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")


@pytest.mark.nightly
@pytest.mark.galaxy_bh
@torch.inference_mode()
def test_streaming_jamba_large_1_6() -> None:
    _setup_logging()
    # Device type must be set before any XLA runtime / SPMD init.
    xr.set_device_type("TT")
    enable_spmd()
    torch.manual_seed(0)
    # Per-layer flush emits one dynamo cache entry per unique (layer, shape);
    # Jamba Large has many hybrid layers and needs more than the default 8.
    torch._dynamo.config.cache_size_limit = 1000

    # torch_xla / PJRT compile options must be strings (see CompilerConfig).
    # With BFP8 weights the device-DRAM ceiling is the bottleneck, so keep the
    # default enable_const_eval_inputs_to_system_memory=true (consteval inputs
    # stay on host) and turn on the DRAM space-saving pass. Without BFP8, force
    # consteval inputs onto device so host RAM stays bounded during staged load.
    compile_opts: Dict[str, str] = {
        "experimental-enable-dram-space-saving-optimization": "true",
    }
    if ENABLE_WEIGHT_BFP8_CONVERSION:
        compile_opts["experimental_weight_dtype"] = "bfp_bf8"
        logger.info(
            "[wdtype] experimental_weight_dtype=bfp_bf8; "
            "consteval inputs on host + DRAM space-saving on"
        )
    else:
        compile_opts["enable_const_eval_inputs_to_system_memory"] = "false"
        logger.info("[wdtype] dense weights; consteval inputs kept on device")
    torch_xla.set_custom_compile_options(compile_opts)

    # EP along the model axis when tt_moe is selected; unused for tt_dense.
    register_tt_moe_backend(cluster_axis=1)

    mesh, mesh_shape = _make_mesh()
    xs.set_global_mesh(mesh)
    device = torch_xla.device()
    num_devices = mesh_shape[0] * mesh_shape[1]
    if BATCH_SIZE % mesh_shape[0] != 0:
        raise ValueError(
            f"BATCH_SIZE ({BATCH_SIZE}) must divide the batch-axis device count "
            f"({mesh_shape[0]}); the batch is sharded on `_axis_0`."
        )

    t_run = time.time()

    _cfg_peek = weight_loader.load_config(MODEL_NAME)
    experts_backend = _select_experts_backend(_cfg_peek.num_experts, mesh_shape)
    logger.info(
        f"[moe] experts_backend={experts_backend} "
        f"(num_experts={_cfg_peek.num_experts}, layers={_cfg_peek.num_hidden_layers}, "
        f"mesh={mesh_shape}, num_devices={num_devices}, "
        f"use_mamba_kernels={_cfg_peek.use_mamba_kernels})"
    )
    del _cfg_peek

    model, config = _build_skeleton(experts_backend)
    layers = list(model.model.layers)
    n_layers = len(layers)
    num_experts = config.num_experts
    if config.num_attention_heads % mesh_shape[1] != 0:
        raise ValueError(
            f"num_attention_heads ({config.num_attention_heads}) must be divisible "
            f"by the model-axis device count ({mesh_shape[1]})."
        )
    if experts_backend == TT_MOE_BACKEND_NAME and num_experts % mesh_shape[1] != 0:
        raise ValueError(
            f"num_experts ({num_experts}) must be divisible by the model-axis "
            f"device count ({mesh_shape[1]}) for tt_moe EP on `_axis_1`."
        )
    _log_mem("baseline")

    prompts_used = [PROMPTS[i % len(PROMPTS)] for i in range(BATCH_SIZE)]
    prompt_ids_cpu, attn_mask_cpu, tok = _tokenize(prompts_used)
    num_decode = 2
    decode_token = prompt_ids_cpu[:, -1:].contiguous()
    decode_tokens_cpu = [decode_token.clone() for _ in range(num_decode)]
    decode_attn_masks_cpu = []
    for k in range(num_decode):
        extra = torch.ones(BATCH_SIZE, k + 1, dtype=attn_mask_cpu.dtype)
        decode_attn_masks_cpu.append(torch.cat([attn_mask_cpu, extra], dim=1))

    # ---- ship top-level params ----
    t = time.time()
    _materialize_and_load(
        model.model.embed_tokens, weight_loader.load_embed_state_dict()
    )
    top_sd = weight_loader.load_top_level_state_dict()
    _materialize_and_load(
        model.model.final_layernorm,
        {"weight": top_sd["model.final_layernorm.weight"]},
    )
    _materialize_and_load(model.lm_head, {"weight": top_sd["lm_head.weight"]})
    del top_sd
    gc.collect()

    h_pref_cpu = model.model.embed_tokens(prompt_ids_cpu).to(torch.bfloat16)
    h_dec_cpu = [
        model.model.embed_tokens(decode_tokens_cpu[k]).to(torch.bfloat16)
        for k in range(num_decode)
    ]
    _cpu = {"h_pref": h_pref_cpu, "h_dec": h_dec_cpu}
    cpu_norm = copy.deepcopy(model.model.final_layernorm)
    cpu_lm_head = copy.deepcopy(model.lm_head)

    def _cpu_logits(h):
        return cpu_lm_head(cpu_norm(h))[:, -1, :]

    pos_pref = (
        torch.arange(PROMPT_LEN, dtype=torch.long).unsqueeze(0).expand(BATCH_SIZE, -1)
    )

    top_spec = _top_level_spec(model)
    for sub in (model.model.embed_tokens, model.model.final_layernorm, model.lm_head):
        _ship_module(sub, top_spec, mesh, device)
    torch_xla.sync(wait=True)
    gc.collect()
    logger.info(f"[step] top-level ship: {time.time() - t:.1f}s")
    _log_mem("post-top-level")

    # ---- dummy flush inputs ----
    h_cpu = torch.zeros(
        BATCH_SIZE, PROMPT_LEN, config.hidden_size, dtype=torch.bfloat16
    )
    h_dummy = _upload(h_cpu, mesh, ("_axis_0", None, None), device)
    pos_dummy = pos_pref.to(device)
    flush_cache = DynamicCache(config=config)
    mask_cpu = _causal_mask(config, h_cpu, attn_mask_cpu, flush_cache, pos_pref)
    mask_dummy = mask_cpu.to(device)
    # Mamba path accepts a 2D pad mask (or None); attention uses the 4D causal mask.
    mamba_mask_dummy = attn_mask_cpu.to(device)
    del h_cpu
    torch_xla.sync(wait=True)

    @torch.compile(backend="tt")
    def run_layer_flush(layer, hidden, attn_mask, position_ids):
        return layer(
            hidden,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
        )

    _t: Dict[str, float] = {}

    def prepare_block(layer, layer_id: int) -> None:
        kind = "attn" if isinstance(layer, JambaAttentionDecoderLayer) else "mamba"
        moe = isinstance(layer.feed_forward, JambaSparseMoeBlock)
        logger.info(
            f"\n[stream] === layer {layer_id}/{n_layers - 1} "
            f"({kind}, moe={moe}) ==="
        )
        _t["start"] = time.time()
        config._experts_implementation = "eager"
        _load_block(layer, layer_id, config)
        gc.collect()
        _t["load"] = time.time() - _t["start"]

    def cpu_golden(layer) -> None:
        """Layer-outer CPU prefill + decode; remap layer_idx→0 for HF cache masks."""
        t0 = time.time()
        cache = DynamicCache()
        # Attention and Mamba both key cache entries by layer_idx.
        mixer = (
            layer.self_attn
            if isinstance(layer, JambaAttentionDecoderLayer)
            else layer.mamba
        )
        saved_layer_idx = mixer.layer_idx
        mixer.layer_idx = 0
        try:
            if isinstance(layer, JambaAttentionDecoderLayer):
                mask_pref = _causal_mask(
                    config, _cpu["h_pref"], attn_mask_cpu, cache, pos_pref
                )
            else:
                # Prefill: attend to all non-pad tokens via 2D mask.
                mask_pref = attn_mask_cpu

            _cpu["h_pref"] = layer(
                _cpu["h_pref"],
                attention_mask=mask_pref,
                position_ids=pos_pref,
                past_key_values=cache,
                use_cache=True,
            )

            for k in range(num_decode):
                past_seen = cache.get_seq_length()
                pos_k = torch.full((BATCH_SIZE, 1), past_seen, dtype=torch.long)
                am = decode_attn_masks_cpu[k]
                if isinstance(layer, JambaAttentionDecoderLayer):
                    mask_k = _causal_mask(config, _cpu["h_dec"][k], am, cache, pos_k)
                else:
                    # Cached mamba forward: mask may be dropped (see _update_mamba_mask).
                    mask_k = None
                _cpu["h_dec"][k] = layer(
                    _cpu["h_dec"][k],
                    attention_mask=mask_k,
                    position_ids=pos_k,
                    past_key_values=cache,
                    use_cache=True,
                )
        finally:
            mixer.layer_idx = saved_layer_idx
        _t["golden"] = time.time() - t0

    def ship_block(layer, layer_id: int) -> None:
        config._experts_implementation = experts_backend
        t_ship = time.time()
        _ship_module(layer, _block_spec(layer, experts_backend), mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        _t["ship"] = time.time() - t_ship

        layer._flush_hook = layer.register_forward_hook(
            sharding_constraint_hook(layer, mesh, ("_axis_0", None, None))
        )

    def post_flush(layer, layer_id: int) -> None:
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        layer._flush_hook.remove()
        del layer._flush_hook
        gc.collect()

        total = time.time() - _t["start"]
        golden = _t.get("golden", 0.0)
        flush = total - _t["load"] - golden - _t["ship"]
        logger.info(
            f"[stream l{layer_id}] total={total:.1f}s load={_t['load']:.1f}s "
            f"golden={golden:.1f}s ship={_t['ship']:.1f}s flush={flush:.1f}s"
        )
        _log_mem(f"l{layer_id} post-flush")

    # ====================== streaming pipeline ======================
    t_loop = time.time()
    for layer_id in range(n_layers):
        layer = layers[layer_id]
        prepare_block(layer, layer_id)
        cpu_golden(layer)
        ship_block(layer, layer_id)
        flush_mask = (
            mask_dummy
            if isinstance(layer, JambaAttentionDecoderLayer)
            else mamba_mask_dummy
        )
        flush_out = run_layer_flush(layer, h_dummy, flush_mask, pos_dummy)
        del flush_out
        post_flush(layer, layer_id)
    logger.info(f"\n[step] per-layer loop: {time.time() - t_loop:.1f}s")

    model.lm_head.register_forward_hook(
        sharding_constraint_hook(model.lm_head, mesh, (None, None))
    )

    logger.info("\n[stream] torch.compile(model) + prefill ...")
    compiled = torch.compile(model, backend="tt")

    prompt_ids = prompt_ids_cpu.to(device)
    xs.mark_sharding(prompt_ids, mesh, ("_axis_0", None))
    attn_mask = attn_mask_cpu.to(device)

    t = time.time()
    past = DynamicCache(config=config)
    prefill_out = compiled(
        input_ids=prompt_ids,
        attention_mask=attn_mask,
        past_key_values=past,
        use_cache=True,
        logits_to_keep=1,
    )
    prefill_logits_dev = prefill_out.logits.to("cpu").float().squeeze(1)
    past = prefill_out.past_key_values
    torch_xla.sync(wait=True)
    logger.info(f"[prefill] compile+exec {time.time() - t:.1f}s")

    decode_logits_dev: List[torch.Tensor] = []
    for k in range(num_decode):
        tok_tt = decode_tokens_cpu[k].to(device)
        xs.mark_sharding(tok_tt, mesh, ("_axis_0", None))
        am = decode_attn_masks_cpu[k].to(device)
        t = time.time()
        out = compiled(
            input_ids=tok_tt,
            attention_mask=am,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        logits = out.logits.to("cpu").float().squeeze(1)
        past = out.past_key_values
        torch_xla.sync(wait=True)
        decode_logits_dev.append(logits)
        kind = "compile+exec" if k == 0 else "exec"
        logger.info(f"[decode {k}] {kind}={time.time() - t:.2f}s")

    prefill_logits_cpu = _cpu_logits(_cpu["h_pref"]).float()
    decode_logits_cpu = [_cpu_logits(h).float() for h in _cpu["h_dec"]]

    logger.info(f"\n[stream] done in {time.time() - t_run:.1f}s\n" + "=" * 72)

    def _topk_agree(cpu_l, dev_l):
        cpu_top1 = cpu_l.argmax(-1)
        top1 = (cpu_top1 == dev_l.argmax(-1)).float().mean().item()
        dev_top5 = dev_l.topk(5, dim=-1).indices
        top5 = (dev_top5 == cpu_top1.unsqueeze(-1)).any(-1).float().mean().item()
        return top1, top5

    rows = [("prefill", prefill_logits_cpu, prefill_logits_dev, PREFILL_PCC_BAR)]
    for k in range(num_decode):
        rows.append(
            (f"decode[{k}]", decode_logits_cpu[k], decode_logits_dev[k], DECODE_PCC_BAR)
        )
    all_pass = True
    for desc, cpu_l, dev_l, bar in rows:
        pcc = compute_pcc(cpu_l, dev_l)
        top1, top5 = _topk_agree(cpu_l, dev_l)
        flag = "" if pcc >= bar else f"  <-- PCC FAIL (<{bar})"
        logger.info(
            f"[pcc] {desc:12s} pcc={pcc:.6f} top1={top1:.3f} top5={top5:.3f}{flag}"
        )
        if pcc < bar:
            all_pass = False
    logger.info("=" * 72)

    # ---- free-run generation ----
    past = DynamicCache(config=config)
    sp_attn = attn_mask_cpu.to(device)
    out = compiled(
        input_ids=prompt_ids,
        attention_mask=sp_attn,
        past_key_values=past,
        use_cache=True,
        logits_to_keep=1,
    )
    next_ids = out.logits.to("cpu").float().squeeze(1).argmax(-1)
    past = out.past_key_values
    generated: List[List[int]] = [[int(next_ids[i])] for i in range(BATCH_SIZE)]
    prev = next_ids.unsqueeze(1)
    cur_mask = torch.cat(
        [attn_mask_cpu, torch.ones(BATCH_SIZE, 1, dtype=attn_mask_cpu.dtype)], dim=1
    )
    for step in range(MAX_NEW_TOKENS - 1):
        prev_tt = prev.to(device)
        xs.mark_sharding(prev_tt, mesh, ("_axis_0", None))
        am = cur_mask.to(device)
        out = compiled(
            input_ids=prev_tt,
            attention_mask=am,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        next_ids = out.logits.to("cpu").float().squeeze(1).argmax(-1)
        past = out.past_key_values
        for i in range(BATCH_SIZE):
            generated[i].append(int(next_ids[i]))
        prev = next_ids.unsqueeze(1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones(BATCH_SIZE, 1, dtype=cur_mask.dtype)], dim=1
        )

    logger.info("[gen] greedy continuations:")
    eos_id = tok.eos_token_id
    for i, ids in enumerate(generated):
        trimmed = ids[: ids.index(eos_id)] if eos_id in ids else ids
        text = tok.decode(trimmed, skip_special_tokens=True)
        logger.info(f"[gen {i:02d}] {prompts_used[i]!r} -> {text!r}")
    logger.info("=" * 72)

    assert (
        all_pass
    ), f"Expected prefill PCC >= {PREFILL_PCC_BAR} and decode PCCs >= {DECODE_PCC_BAR}"
