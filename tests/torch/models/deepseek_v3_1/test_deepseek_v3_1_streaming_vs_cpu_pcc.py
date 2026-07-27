# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY: PCC comparison for the DeepSeek-V3.1 streaming path (4 layers, b128).

Streams the model onto device exactly like
``test_deepseek_v3_1_e2e_streaming.py`` (reusing its helpers) but ALSO builds a
CPU-bf16 reference with the V4 "layer-outer" trick: while each block's weights
are still on host (just after loading, before shipping), the CPU prefill + a few
teacher-forced decode hidden states are advanced through that block, accumulating
into a CPU MLA cache. Peak host RAM therefore stays bounded to ~one block -- no
second full model is ever held -- so this stays valid at higher depths too.

Both sides are fed IDENTICAL teacher-forced inputs at every step, so the logits
are directly comparable. We assert PCC between the streamed-device logits and the
CPU reference. This is the rigorous version of the print+eyeball smoke test: the
4-layer output is incoherent, but PCC compares the continuous logit vectors (the
computed function), which is robust to the argmax tie-breaking noise that makes
individual rows diverge.

Delete this file once the streaming path is trusted -- it is not meant to live in
the suite long-term.

    pytest -svv tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_streaming_vs_cpu_pcc.py
"""

from __future__ import annotations

import copy
import gc
import os
import time

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger

from . import weight_loader
from .test_deepseek_v3_1_e2e_streaming import (
    INPUT_PROMPT,
    _alias_rotary_across_layers,
    _block_shard_spec,
    _init_mla_cache,
    _log_mem,
    _make_mesh,
    _materialize_rotary_caches,
    _reinit_cache_buffers,
    _setup_logging,
    _ship_module,
    _strip_cpu_golden,
    _top_level_shard_spec,
    _transfer_and_shard_cache,
)

# Hard-coded here on purpose, independent of the main streaming test, so you can
# freely iterate on NUM_LAYERS / BATCH_SIZE there without perturbing this PCC
# comparison (and vice-versa). This test also uses its own _build_skeleton below
# so it does not pick up the main module's NUM_LAYERS.
NUM_LAYERS = 4
BATCH_SIZE = 128
# Teacher-forced decode steps to PCC-check (in addition to prefill).
NUM_DECODE = 2
# Prefill drifts a bit more than a single decode step; keep both bars meaningful
# but not brittle. The non-streaming benchmark hits ~0.99 at 4 layers.
PREFILL_PCC_BAR = 0.95
DECODE_PCC_BAR = 0.95


def _block_out(out):
    """DeepseekV3DecoderLayer.forward returns a tuple whose [0] is hidden_states."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _build_skeleton(loader):
    """Weight-less DeepSeek-V3.1 (this file's NUM_LAYERS) built on meta. Local copy
    so it uses this test's NUM_LAYERS rather than the main module's."""
    from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.modified_modeling_deepseek import (
        DeepseekV3ForCausalLM,
    )

    config = loader._load_config(num_layers=NUM_LAYERS)
    config._experts_implementation = "batched_mm"
    config.layer_types = ["full_attention"] * NUM_LAYERS
    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config).eval()
    return model, config


@pytest.mark.nightly
@pytest.mark.galaxy
@torch.inference_mode()
def test_streaming_vs_cpu_pcc() -> None:
    from tt_torch.sharding import sharding_constraint_hook
    from tt_torch.sparse_mlp import enable_sparse_mlp

    from tests.benchmark.utils import compute_pcc
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
    torch._dynamo.config.cache_size_limit = 1000
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": 0,
            "enable_trace": False,
            "enable_const_eval_inputs_to_system_memory": False,
        }
    )

    device = torch_xla.device()
    loader = ModelLoader(
        variant=ModelVariant.DEEPSEEK_V3_1_MODIFIED, num_layers=NUM_LAYERS
    )
    mesh, mesh_shape = _make_mesh(loader)
    if BATCH_SIZE % mesh_shape[0] != 0:
        raise ValueError(f"BATCH_SIZE {BATCH_SIZE} must divide batch axis {mesh_shape[0]}")

    t_run = time.time()

    # ---- teacher-forced inputs (identical for CPU + device) ----
    tokenizer = loader._load_tokenizer()
    prompt_ids_1 = tokenizer(INPUT_PROMPT, return_tensors="pt")["input_ids"][0]
    prompt_len = int(prompt_ids_1.shape[0])
    prompt_ids_cpu = prompt_ids_1.unsqueeze(0).expand(BATCH_SIZE, -1).contiguous()
    # Reuse prompt tokens as teacher-forced decode inputs (any valid tokens work;
    # CPU and device see the same ones so their logits stay comparable).
    decode_ids_cpu = [
        prompt_ids_cpu[:, k : k + 1].contiguous() for k in range(NUM_DECODE)
    ]
    needed = prompt_len + NUM_DECODE
    max_cache_len = ((needed + 31) // 32) * 32
    logger.info(
        f"[cfg] prompt_len={prompt_len} num_decode={NUM_DECODE} "
        f"max_cache_len={max_cache_len} batch={BATCH_SIZE} layers={NUM_LAYERS}"
    )

    # ---- skeleton + rotary ----
    model, config = _build_skeleton(loader)
    layers = model.model.layers
    n_layers = len(layers)
    _alias_rotary_across_layers(model)
    _materialize_rotary_caches(model, seq_len=max_cache_len, dtype=torch.bfloat16)

    # ---- top-level: load on CPU, embed inputs + snapshot norm/head on CPU, ship ----
    top_sd = weight_loader.load_top_level_state_dict(loader._BF16_WEIGHTS_REPO)
    _, unexpected = model.load_state_dict(top_sd, strict=False, assign=True)
    if unexpected:
        raise RuntimeError(f"top-level load: unexpected {sorted(unexpected)[:8]}")
    del top_sd
    gc.collect()

    # CPU golden references (captured before shipping moves them to device).
    cpu_embed = model.model.embed_tokens
    h_pref_cpu = cpu_embed(prompt_ids_cpu)  # (B, prompt_len, H) bf16
    h_dec_cpu = [cpu_embed(decode_ids_cpu[k]) for k in range(NUM_DECODE)]
    cpu_norm = copy.deepcopy(model.model.norm)
    cpu_lm_head = copy.deepcopy(model.lm_head)

    top_spec_by_id = {id(t): s for t, s in _top_level_shard_spec(model).items()}
    _ship_module(model.model.embed_tokens, top_spec_by_id, mesh, device)
    _ship_module(model.model.norm, top_spec_by_id, mesh, device)
    _ship_module(model.lm_head, top_spec_by_id, mesh, device)
    model.lm_head.register_forward_hook(
        sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
    )
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()

    # ---- CPU MLA cache (golden) + device MLA cache ----
    cpu_cache = _init_mla_cache(config, BATCH_SIZE, max_cache_len)  # stays on CPU
    dev_cache = _init_mla_cache(config, BATCH_SIZE, max_cache_len)
    _transfer_and_shard_cache(dev_cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()

    # ---- device per-block dummy-flush plumbing (q_len=1) ----
    dummy_hidden = torch.zeros(
        BATCH_SIZE, 1, config.hidden_size, dtype=torch.bfloat16
    ).to(device)
    xs.mark_sharding(dummy_hidden, mesh, ("batch", None, None))
    flush_cache_pos_cpu = torch.tensor([0], dtype=torch.long)
    dummy_mask = (
        dev_cache.layers[0]
        .build_causal_mask(flush_cache_pos_cpu, BATCH_SIZE, torch.bfloat16, torch.device("cpu"))
        .to(device)
    )
    xs.mark_sharding(dummy_mask, mesh, ("batch", None, None, None))
    dummy_pos_ids = torch.tensor([[0]], dtype=torch.long).to(device)
    dummy_cache_pos = flush_cache_pos_cpu.to(device)
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

    # ---- CPU-golden step helper (advance one hidden state through one block) ----
    def _cpu_step(block, hidden, cache_pos_cpu):
        mask = cpu_cache.layers[0].build_causal_mask(
            cache_pos_cpu, BATCH_SIZE, torch.bfloat16, torch.device("cpu")
        )
        out = block(
            hidden,
            attention_mask=mask,
            position_ids=cache_pos_cpu.unsqueeze(0),
            past_key_value=cpu_cache,
            use_cache=True,
            cache_position=cache_pos_cpu,
        )
        return _block_out(out)

    cp_prefill = torch.arange(0, prompt_len, dtype=torch.long)
    cp_decode = [torch.tensor([prompt_len + k], dtype=torch.long) for k in range(NUM_DECODE)]

    # ---- per-block: load -> CPU golden (dense) -> sparse -> ship -> flush -> free ----
    logger.info(f"[stream] streaming {n_layers} block(s) (dummy_flush={do_flush}) ...")
    t_loop = time.time()
    for i in range(n_layers):
        t_blk = time.time()
        block = layers[i]

        block_sd = weight_loader.load_block_state_dict(loader._BF16_WEIGHTS_REPO, i)
        _, unexpected = block.load_state_dict(block_sd, strict=False, assign=True)
        if unexpected:
            raise RuntimeError(f"layer {i} load: unexpected {sorted(unexpected)[:8]}")
        del block_sd
        gc.collect()

        # CPU golden through this (still-dense, CPU-resident) block, time-ordered
        # so its CPU KV cache is built before the weights leave host.
        h_pref_cpu = _cpu_step(block, h_pref_cpu, cp_prefill)
        for k in range(NUM_DECODE):
            h_dec_cpu[k] = _cpu_step(block, h_dec_cpu[k], cp_decode[k])

        # Device path: sparse-MLP swap (no-op for dense layers 0-2), drop the dense
        # CPU-golden copies it keeps (our CPU reference already ran above), ship, flush.
        enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=config)
        _strip_cpu_golden(block)
        gc.collect()
        block_spec_by_id = {id(t): s for t, s in _block_shard_spec(block).items()}
        _ship_module(block, block_spec_by_id, mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        if do_flush:
            run_block_flush(
                block, dummy_hidden, dummy_mask, dummy_pos_ids, dev_cache, dummy_cache_pos
            )
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
        gc.collect()
        logger.info(
            f"[stream l{i}] {time.time() - t_blk:.1f}s (moe={type(block.mlp).__name__})"
        )
        _log_mem(f"l{i} post-flush")
    logger.info(f"[step] per-block loop: {time.time() - t_loop:.1f}s")

    # ---- CPU reference logits (norm + head on the streamed CPU hidden states) ----
    def _cpu_logits(h):
        return cpu_lm_head(cpu_norm(h)).float()

    ref_prefill = _cpu_logits(h_pref_cpu)
    ref_decode = [_cpu_logits(h_dec_cpu[k]) for k in range(NUM_DECODE)]

    # ---- device run: same teacher-forced inputs, fresh cache ----
    _reinit_cache_buffers(dev_cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()

    logger.info("[stream] torch.compile(model) + device prefill/decode ...")
    compiled = torch.compile(model, backend="tt")

    prompt_ids = prompt_ids_cpu.to(device)
    xs.mark_sharding(prompt_ids, mesh, ("batch", None))
    cp = torch.arange(0, prompt_len, dtype=torch.long).to(device)
    t = time.time()
    dev_prefill = (
        compiled(
            input_ids=prompt_ids,
            past_key_values=dev_cache,
            cache_position=cp,
            use_cache=True,
        ).logits.to("cpu").float()
    )
    torch_xla.sync(wait=True)
    logger.info(f"[prefill] compile+exec {time.time() - t:.1f}s shape={tuple(dev_prefill.shape)}")

    dev_decode = []
    for k in range(NUM_DECODE):
        tok = decode_ids_cpu[k].to(device)
        xs.mark_sharding(tok, mesh, ("batch", None))
        cpk = torch.tensor([prompt_len + k], dtype=torch.long).to(device)
        t = time.time()
        out = compiled(
            input_ids=tok,
            past_key_values=dev_cache,
            cache_position=cpk,
            use_cache=True,
        ).logits.to("cpu").float()
        torch_xla.sync(wait=True)
        dev_decode.append(out)
        logger.info(f"[decode {k}] sp={prompt_len + k} {time.time() - t:.2f}s")

    # ---- PCC ----
    logger.info(f"[stream] done in {time.time() - t_run:.1f}s\n" + "=" * 72)

    def _topk_agree(cpu_l, dev_l):
        cpu_top1 = cpu_l.argmax(-1)
        top1 = (cpu_top1 == dev_l.argmax(-1)).float().mean().item()
        dev_top5 = dev_l.topk(5, dim=-1).indices
        top5 = (dev_top5 == cpu_top1.unsqueeze(-1)).any(-1).float().mean().item()
        return top1, top5

    rows = [("prefill", ref_prefill, dev_prefill, PREFILL_PCC_BAR)]
    for k in range(NUM_DECODE):
        rows.append((f"decode[{k}]", ref_decode[k], dev_decode[k], DECODE_PCC_BAR))

    all_pass = True
    lines = ["[pcc] device(streamed) vs CPU golden:"]
    for name, ref, dev, bar in rows:
        pcc = compute_pcc(ref, dev)
        top1, top5 = _topk_agree(ref, dev)
        flag = "" if pcc >= bar else f"  <-- FAIL (<{bar})"
        lines.append(
            f"  {name:11s} pcc={pcc:.6f} top1={top1:.3f} top5={top5:.3f}{flag}"
        )
        if pcc < bar:
            all_pass = False
    logger.info("\n".join(lines))
    logger.info("=" * 72)

    assert all_pass, (
        f"Streaming device logits diverge from the CPU reference "
        f"(prefill bar {PREFILL_PCC_BAR}, decode bar {DECODE_PCC_BAR})."
    )
