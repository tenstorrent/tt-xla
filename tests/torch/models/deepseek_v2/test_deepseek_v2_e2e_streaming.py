# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Streaming-inference end-to-end test for DeepSeek-V2-Chat on Tenstorrent.

DeepSeek-V2-Chat is a 236B-parameter MoE (60 layers, 160 routed experts) —
too large to hold in host RAM all at once. We stream it the same way as the
DeepSeek-V4-Flash streaming e2e:

    build meta skeleton (no real weight storage)
    ship top-level params (embed / norm / lm_head)
    for layer in model.model.layers:
        load HF weights → swap MoE → ship sharded → dummy-flush
    model = torch.compile(model, backend="tt")
    prefill + decode loop

Unlike V4's custom decode-opt stack (registered KV buffers + start_pos), this
uses the HuggingFace DeepseekV2ForCausalLM path (MLA + Dynamic/tuple cache).
CPU-reference PCC still runs layer-outer while each block's weights are on
host; the device run is time-outer with teacher-forced decode inputs.

    pytest -svv tests/torch/models/deepseek_v2/test_deepseek_v2_e2e_streaming.py
"""

from __future__ import annotations

import copy
import gc
import logging
import sys
import time
import warnings
from typing import Dict, List, Optional, Tuple

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
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from tt_torch.sharding import sharding_constraint_hook, sharding_constraint_tensor
from tt_torch.sparse_mlp import enable_sparse_mlp
from ttxla_tools.logging import logger

from tests.benchmark.utils import compute_pcc

from . import weight_loader

# ---- run configuration (intentionally fixed for this bring-up) ----
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Chat"
# Must divide mesh axis-0 (4 on Galaxy / 2 on 8-chip). Same prompt is repeated.
BATCH_SIZE = 4
MAX_NEW_TOKENS = 16
PROMPT_LEN = 128
# Full V2-Chat is 60 layers. Streaming keeps *all* shipped weights on device.
# BF16 OOMs ~layer 20 on Galaxy; BFP8 (~2x smaller const-eval weights) is the
# path toward the full stack. Set to an int to cap layers for bring-up.
NUM_LAYERS = None
# Same as runner YAML ``enable_weight_bfp8_conversion: true`` →
# compiler_config.experimental_weight_dtype = "bfp_bf8".
ENABLE_WEIGHT_BFP8_CONVERSION = True

PREFILL_PCC_BAR = 0.91
DECODE_PCC_BAR = 0.95

PROMPTS = [
    "What is the capital of France?",
]


# ---------------------------------------------------------------------------
# Host-memory diagnostics
# ---------------------------------------------------------------------------
def _log_mem(tag: str) -> None:
    """Log host RSS so the bounded per-layer footprint is visible."""
    import os

    import psutil

    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    sys_used = psutil.virtual_memory().used / 1e9
    logger.info(f"[mem {tag:24s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")


# ---------------------------------------------------------------------------
# Mesh + sharded upload helpers
# ---------------------------------------------------------------------------
def _make_mesh() -> Tuple[Mesh, Tuple[int, int], int]:
    """2D device mesh matching V4 streaming / deepseek_moe loader.

    axis 0 = batch (data) parallel, axis 1 = model/tensor.
    cluster_axis=0 so sparse-MLP dispatch runs along the batch mesh dim.
    """
    n = xr.global_runtime_device_count()
    if n == 32:
        mesh_shape = (4, 8)
    elif n == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, n)
    cluster_axis = 0
    logger.info(
        f"[mesh] num_devices={n} mesh_shape={mesh_shape} cluster_axis={cluster_axis}"
    )
    mesh = Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1"))
    return mesh, mesh_shape, cluster_axis



def _upload(cpu_tensor: torch.Tensor, mesh, partition_spec, device) -> torch.Tensor:
    """Move a CPU tensor to the XLA device and annotate its shard spec."""
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
    sharded copy. Drops the source CPU tensors so the caller can gc them."""
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
# Model construction + per-layer sharding specs (DeepSeek-V2-Chat)
# ---------------------------------------------------------------------------
def _patch_transformers_for_dsv2_remote_code() -> None:
    """Bridge HF DeepSeek-V2-Chat remote code onto newer transformers.

    The Hub ``modeling_deepseek.py`` still expects:
      * ``is_torch_fx_available`` (removed from import_utils)
      * ``Cache.get_usable_length`` (renamed to ``get_seq_length``)
      * ``DynamicCache.from_legacy_cache`` / ``to_legacy_cache`` (removed)
    """
    import transformers.utils.import_utils as import_utils
    from transformers.cache_utils import Cache, DynamicCache

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: False

    if not hasattr(Cache, "get_usable_length"):

        def get_usable_length(self, seq_length=None, layer_idx: int = 0):
            # Old API ignored seq_length and returned cached KV length.
            return self.get_seq_length(layer_idx)

        Cache.get_usable_length = get_usable_length

    if not hasattr(DynamicCache, "from_legacy_cache"):

        @classmethod
        def from_legacy_cache(cls, past_key_values=None):
            cache = cls()
            if past_key_values is None:
                return cache
            for layer_idx, layer in enumerate(past_key_values):
                if layer is None:
                    continue
                key_states, value_states = layer[0], layer[1]
                cache.update(key_states, value_states, layer_idx)
            return cache

        DynamicCache.from_legacy_cache = from_legacy_cache

    if not hasattr(DynamicCache, "to_legacy_cache"):

        def to_legacy_cache(self):
            legacy = []
            for layer in self.layers:
                if not getattr(layer, "is_initialized", False) or layer.keys.numel() == 0:
                    legacy.append(None)
                else:
                    legacy.append((layer.keys, layer.values))
            return tuple(legacy)

        DynamicCache.to_legacy_cache = to_legacy_cache


def _rematerialize_rotary_on_cpu(model: nn.Module) -> None:
    """Rebuild rotary modules after meta-init (non-persistent buffers stay meta).

    DeepseekV2Attention exposes ``_init_rope``; re-running it under
    ``torch.device("cpu")`` materializes ``inv_freq`` / cos/sin caches.
    """
    with torch.device("cpu"):
        for module in model.modules():
            if hasattr(module, "_init_rope"):
                module._init_rope()


def _build_skeleton():
    """Empty (meta-weight) DeepseekV2ForCausalLM in bf16."""
    _patch_transformers_for_dsv2_remote_code()
    config = weight_loader.load_config(MODEL_NAME)
    if NUM_LAYERS is not None:
        config.num_hidden_layers = NUM_LAYERS
        logger.info(
            f"[skel] capping num_hidden_layers={NUM_LAYERS} "
            f"(full checkpoint has more; Galaxy DRAM bring-up)"
        )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model.eval()
    _rematerialize_rotary_on_cpu(model)
    return model, config


def _top_level_spec(model) -> Dict[int, Tuple]:
    """embed / norm replicated; lm_head column-sharded on the model axis."""
    spec = {
        model.model.embed_tokens.weight: (None, None),
        model.model.norm.weight: (None,),
        model.lm_head.weight: ("_axis_1", None),
    }
    return {id(t): ps for t, ps in spec.items()}


def _block_spec(layer, mesh_shape) -> Dict[int, Tuple]:
    """SPMD shard spec for one HF decoder layer (MLA + dense/MoE MLP)."""
    compound = ("_axis_0", "_axis_1")
    specs: Dict[torch.Tensor, Tuple] = {}

    sa = layer.self_attn
    # Q-LoRA path (V2-Chat): q_a replicated latent, q_b / kv_b column-sharded.
    if getattr(sa, "q_lora_rank", None) is not None:
        specs[sa.q_a_proj.weight] = (None, None)
        specs[sa.q_b_proj.weight] = ("_axis_1", None)
    else:
        specs[sa.q_proj.weight] = ("_axis_1", None)
    specs[sa.kv_a_proj_with_mqa.weight] = (None, None)
    specs[sa.kv_b_proj.weight] = ("_axis_1", None)
    specs[sa.o_proj.weight] = (None, "_axis_1")

    mlp = layer.mlp
    # After enable_sparse_mlp, MoE layers become A2aSparseMLPWithSharedExperts.
    if hasattr(mlp, "mlp") and hasattr(mlp.mlp, "experts"):
        a2a = mlp.mlp
        specs[a2a.router.gate.weight] = (None, None)
        specs[a2a.experts.gate_proj] = (compound, None, None)
        specs[a2a.experts.up_proj] = (compound, None, None)
        specs[a2a.experts.down_proj] = (compound, None, None)
        shared = getattr(mlp, "shared_experts", None)
        if shared is not None:
            specs[shared.gate_proj.weight] = (None, None)
            specs[shared.up_proj.weight] = (None, None)
            specs[shared.down_proj.weight] = (None, None)
    elif hasattr(mlp, "experts"):
        # Dense MoE before sparse swap (should not ship in this state).
        pass
    else:
        # Layer-0 dense MLP.
        specs[mlp.gate_proj.weight] = ("_axis_1", None)
        specs[mlp.up_proj.weight] = ("_axis_1", None)
        specs[mlp.down_proj.weight] = (None, "_axis_1")

    return {id(t): ps for t, ps in specs.items()}


def _strip_cpu_golden(layer) -> None:
    """Drop dense-MoE CPU references kept by enable_sparse_mlp (~GBs/layer)."""
    mlp = layer.mlp
    a2a = getattr(mlp, "mlp", None)
    if a2a is None:
        return
    if hasattr(a2a, "_original_mlp"):
        object.__setattr__(a2a, "_original_mlp", None)
    experts = getattr(a2a, "experts", None)
    if experts is not None and "original_experts" in getattr(experts, "_modules", {}):
        del experts._modules["original_experts"]


def _load_block(layer, layer_id: int) -> None:
    """Load layer `layer_id`'s HF weights into `layer` (meta → CPU assign)."""
    sd = weight_loader.load_block_state_dict(MODEL_NAME, layer_id)
    result = layer.load_state_dict(sd, strict=False, assign=True)
    if result.unexpected_keys:
        raise RuntimeError(
            f"layers.{layer_id}: unexpected keys {sorted(result.unexpected_keys)}"
        )
    # Any remaining meta params (e.g. unused) stay meta; rotary already on CPU.
    still_meta = [
        n for n, p in layer.named_parameters() if p.device.type == "meta"
    ]
    if still_meta:
        raise RuntimeError(
            f"layers.{layer_id}: still-meta params after load: {still_meta[:8]}"
        )
    del sd


def _tokenize(prompts: List[str]) -> Tuple[torch.Tensor, object]:
    """Apply the DeepSeek-V2 chat template and left-pad to PROMPT_LEN."""
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    rows = []
    for i in range(BATCH_SIZE):
        messages = [{"role": "user", "content": prompts[i % len(prompts)]}]
        # DeepSeek's remote tokenizer often returns the rendered string even when
        # tokenize=True is passed; render then encode explicitly.
        text = tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        if ids.shape[0] >= PROMPT_LEN:
            ids = ids[-PROMPT_LEN:]
        else:
            ids = torch.cat(
                [
                    torch.full((PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long),
                    ids,
                ]
            )
        rows.append(ids)
    return torch.stack(rows, dim=0).contiguous(), tok


def _causal_mask(
    bsz: int, q_len: int, kv_len: int, dtype: torch.dtype
) -> torch.Tensor:
    """Additive causal mask [B, 1, q_len, kv_len] (0 attend, -inf masked)."""
    mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=dtype)
    # Position i (in the full kv timeline) may attend to j <= i.
    q_offset = kv_len - q_len
    for i in range(q_len):
        mask[:, :, i, (q_offset + i + 1) :] = torch.finfo(dtype).min
    return mask


def _layer_output_sharding_hook(mesh, partition_spec):
    """Like sharding_constraint_hook, but HF decoder layers return tuples."""

    def hook(mod, inputs, output):
        if isinstance(output, tuple):
            hs = sharding_constraint_tensor(output[0], mesh, partition_spec)
            return (hs,) + output[1:]
        return sharding_constraint_tensor(output, mesh, partition_spec)

    return hook


# ---------------------------------------------------------------------------
# Main streaming pipeline
# ---------------------------------------------------------------------------
def _setup_logging() -> None:
    warnings.filterwarnings("ignore")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {message}")


@pytest.mark.nightly
@pytest.mark.bh_galaxy
@torch.inference_mode()
def test_streaming_dsv2_chat() -> None:
    _setup_logging()
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)
    # Per-layer flush emits one dynamo cache entry per unique (layer, shape);
    # 60 layers needs more than the default 8.
    torch._dynamo.config.cache_size_limit = 1000
    compile_opts = {"enable_const_eval_inputs_to_system_memory": False}
    if ENABLE_WEIGHT_BFP8_CONVERSION:
        # Same compile flag the runner sets for enable_weight_bfp8_conversion.
        compile_opts["experimental_weight_dtype"] = "bfp_bf8"
        logger.info("[wdtype] experimental_weight_dtype=bfp_bf8 (global)")
    torch_xla.set_custom_compile_options(compile_opts)

    mesh, mesh_shape, cluster_axis = _make_mesh()
    device = torch_xla.device()
    if BATCH_SIZE % mesh_shape[0] != 0:
        raise ValueError(
            f"BATCH_SIZE ({BATCH_SIZE}) must divide the batch-axis device count "
            f"({mesh_shape[0]}); the batch is sharded on `_axis_0`."
        )

    t_run = time.time()

    # ---- skeleton (meta weights) ----
    model, config = _build_skeleton()
    layers = list(model.model.layers)
    n_layers = len(layers)
    hidden_size = config.hidden_size
    _log_mem("baseline")

    prompts_used = [PROMPTS[i % len(PROMPTS)] for i in range(BATCH_SIZE)]
    prompt_ids_cpu, tok = _tokenize(prompts_used)
    # Two teacher-forced decode steps: covers decode branch + KV continuity.
    num_decode = 2
    decode_tokens_cpu = [
        prompt_ids_cpu[:, k : k + 1].contiguous() for k in range(num_decode)
    ]

    # ---- ship top-level params (embed / norm / lm_head) ----
    t = time.time()
    embed_sd = weight_loader.load_embed_state_dict(MODEL_NAME)
    model.model.embed_tokens.load_state_dict(embed_sd, strict=False, assign=True)
    del embed_sd
    model.load_state_dict(
        weight_loader.load_top_level_state_dict(MODEL_NAME), strict=False, assign=True
    )
    gc.collect()

    # Embed reference hidden states on host; keep CPU copies of head stack.
    h_pref_cpu = model.model.embed_tokens(prompt_ids_cpu).to(torch.bfloat16)
    h_dec_cpu = [
        model.model.embed_tokens(decode_tokens_cpu[k]).to(torch.bfloat16)
        for k in range(num_decode)
    ]
    _cpu = {"h_pref": h_pref_cpu, "h_dec": h_dec_cpu}
    # Per-layer past_key_value for the CPU layer-outer golden path.
    _cpu_past: List[Optional[Tuple]] = [None] * n_layers

    cpu_lm_head = copy.deepcopy(model.lm_head).to("cpu")
    cpu_norm = copy.deepcopy(model.model.norm).to("cpu")

    def _cpu_logits(h):
        return cpu_lm_head(cpu_norm(h))

    top_spec = _top_level_spec(model)
    _ship_module(model.model.embed_tokens, top_spec, mesh, device)
    _ship_module(model.model.norm, top_spec, mesh, device)
    _ship_module(model.lm_head, top_spec, mesh, device)
    torch_xla.sync(wait=True)
    gc.collect()
    logger.info(f"[step] top-level ship: {time.time() - t:.1f}s")
    _log_mem("post-top-level")

    # ---- dummy block inputs (real prefill shape) for the per-layer flush ----
    h_cpu = torch.zeros(BATCH_SIZE, PROMPT_LEN, hidden_size, dtype=torch.bfloat16)
    h_dummy = _upload(h_cpu, mesh, ("_axis_0", None, None), device)
    pos_dummy = torch.arange(PROMPT_LEN, dtype=torch.long).unsqueeze(0).to(device)
    attn_dummy = _causal_mask(BATCH_SIZE, PROMPT_LEN, PROMPT_LEN, torch.bfloat16).to(
        device
    )
    del h_cpu
    torch_xla.sync(wait=True)

    @torch.compile(backend="tt")
    def run_block_flush(layer, hidden_states, attention_mask, position_ids):
        out = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        return out[0] if isinstance(out, tuple) else out

    _t: Dict[str, float] = {}

    def prepare_block(layer, layer_id: int) -> None:
        """Load HF weights and swap dense MoE for the sparse version."""
        is_moe = hasattr(layer.mlp, "experts")
        logger.info(
            f"\n[stream] === layer {layer_id}/{n_layers - 1} "
            f"({'moe' if is_moe else 'dense'}) ==="
        )
        _t["start"] = time.time()

        _load_block(layer, layer_id)
        gc.collect()
        if is_moe:
            enable_sparse_mlp(
                layer,
                mesh=mesh_shape,
                cluster_axis=cluster_axis,
                config=config,
                verbose=False,
            )
            gc.collect()
        _t["load"] = time.time() - _t["start"]

    def cpu_golden(layer, layer_id: int) -> None:
        """Advance CPU prefill + decode hidden states through this layer."""
        t0 = time.time()
        # DeepSeek-V2 attention only writes into an existing Cache; passing None
        # with use_cache=True returns None and decode then RoPE-indexes OOB.
        past = _cpu_past[layer_id]
        if past is None:
            past = DynamicCache()

        attn_pref = _causal_mask(
            BATCH_SIZE, PROMPT_LEN, PROMPT_LEN, _cpu["h_pref"].dtype
        )
        pos_pref = torch.arange(PROMPT_LEN, dtype=torch.long).unsqueeze(0)
        out = layer(
            _cpu["h_pref"],
            attention_mask=attn_pref,
            position_ids=pos_pref,
            past_key_value=past,
            use_cache=True,
        )
        _cpu["h_pref"] = out[0]
        past = out[1] if len(out) > 1 else past

        for k in range(num_decode):
            kv_len = PROMPT_LEN + k + 1
            attn_dec = _causal_mask(BATCH_SIZE, 1, kv_len, _cpu["h_dec"][k].dtype)
            pos_dec = torch.tensor([[PROMPT_LEN + k]], dtype=torch.long)
            out = layer(
                _cpu["h_dec"][k],
                attention_mask=attn_dec,
                position_ids=pos_dec,
                past_key_value=past,
                use_cache=True,
            )
            _cpu["h_dec"][k] = out[0]
            past = out[1] if len(out) > 1 else past

        _cpu_past[layer_id] = past
        _t["golden"] = time.time() - t0

    def ship_block(layer, layer_id: int) -> None:
        """Drop CPU golden refs, ship sparse/dense weights, arm output shard hook."""
        _strip_cpu_golden(layer)
        gc.collect()
        t_ship = time.time()
        _ship_module(layer, _block_spec(layer, mesh_shape), mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        _t["ship"] = time.time() - t_ship

        layer._flush_hook = layer.register_forward_hook(
            _layer_output_sharding_hook(mesh, ("_axis_0", None, None))
        )

    def post_flush(layer, layer_id: int) -> None:
        """Await dummy flush (migrates plugin staging off host), drop hook, log."""
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

    # ====================== the streaming pipeline ======================
    t_loop = time.time()
    for layer_id in range(n_layers):
        layer = layers[layer_id]
        prepare_block(layer, layer_id)
        cpu_golden(layer, layer_id)
        ship_block(layer, layer_id)
        run_block_flush(layer, h_dummy, attn_dummy, pos_dummy)
        post_flush(layer, layer_id)
    logger.info(f"\n[step] per-layer loop: {time.time() - t_loop:.1f}s")

    # Keep lm_head logits replicated across devices after the gather.
    model.lm_head.register_forward_hook(
        sharding_constraint_hook(model.lm_head, mesh, (None, None))
    )

    # ---- whole-model compile, then teacher-forced prefill + decode ----
    logger.info("\n[stream] torch.compile(model) + prefill ...")
    compiled = torch.compile(model, backend="tt")

    prompt_ids = prompt_ids_cpu.to(device)
    xs.mark_sharding(prompt_ids, mesh, ("_axis_0", None))
    attn_pref = (
        (prompt_ids_cpu != (tok.pad_token_id or tok.eos_token_id))
        .to(torch.long)
        .to(device)
    )

    t = time.time()
    pref_out = compiled(
        input_ids=prompt_ids,
        attention_mask=attn_pref,
        use_cache=True,
        return_dict=True,
    )
    prefill_logits_dev = pref_out.logits[:, -1, :].to("cpu").float()
    past = pref_out.past_key_values
    torch_xla.sync(wait=True)
    logger.info(f"[prefill] compile+exec {time.time() - t:.1f}s")

    decode_logits_dev: List[torch.Tensor] = []
    for k in range(num_decode):
        tok_tt = decode_tokens_cpu[k].to(device)
        xs.mark_sharding(tok_tt, mesh, ("_axis_0", None))
        # Extend the 2D attention mask by one attending position.
        ones = torch.ones(BATCH_SIZE, 1, dtype=attn_pref.dtype, device=device)
        attn_pref = torch.cat([attn_pref, ones], dim=1)
        t = time.time()
        dec_out = compiled(
            input_ids=tok_tt,
            attention_mask=attn_pref,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        logits = dec_out.logits[:, -1, :].to("cpu").float()
        past = dec_out.past_key_values
        torch_xla.sync(wait=True)
        decode_logits_dev.append(logits)
        kind = "compile+exec" if k == 0 else "exec"
        logger.info(f"[decode {k}] {kind}={time.time() - t:.2f}s")

    # ---- CPU reference logits (from the layer-outer streaming pass) ----
    prefill_logits_cpu = _cpu_logits(_cpu["h_pref"])[:, -1, :].float()
    decode_logits_cpu = [_cpu_logits(h)[:, -1, :].float() for h in _cpu["h_dec"]]

    # ---- PCC + top-k agreement ----
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

    # ---- free-run generation so output coherence is visible in CI ----
    # Rebuild a clean past via a fresh prefill (teacher-forced path left cache dirty
    # with forced tokens that are not the model's own predictions).
    t = time.time()
    gen_attn = (
        (prompt_ids_cpu != (tok.pad_token_id or tok.eos_token_id))
        .to(torch.long)
        .to(device)
    )
    gen_out = compiled(
        input_ids=prompt_ids,
        attention_mask=gen_attn,
        use_cache=True,
        return_dict=True,
    )
    next_ids = gen_out.logits[:, -1, :].to("cpu").float().argmax(-1)
    past = gen_out.past_key_values
    generated: List[List[int]] = [[int(next_ids[i])] for i in range(BATCH_SIZE)]
    prev = next_ids.unsqueeze(1)
    for step in range(MAX_NEW_TOKENS - 1):
        prev_tt = prev.to(device)
        xs.mark_sharding(prev_tt, mesh, ("_axis_0", None))
        ones = torch.ones(BATCH_SIZE, 1, dtype=gen_attn.dtype, device=device)
        gen_attn = torch.cat([gen_attn, ones], dim=1)
        gen_out = compiled(
            input_ids=prev_tt,
            attention_mask=gen_attn,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        next_ids = gen_out.logits[:, -1, :].to("cpu").float().argmax(-1)
        past = gen_out.past_key_values
        for i in range(BATCH_SIZE):
            generated[i].append(int(next_ids[i]))
        prev = next_ids.unsqueeze(1)
    logger.info(f"[gen] greedy loop {time.time() - t:.1f}s")

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
