# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TEMPORARY: per-layer streaming memory / correctness validation.

Streams a small stack (3 dense + 1 MoE layer) one block at a time and, for EACH
block, checks that the streaming infra:

  1. moved everything off the host  -- after ship+flush the block holds ZERO
     CPU-resident parameters/buffers (host is cleared);
  2. kept nothing extra              -- the dense CPU-golden copies
     (`_original_mlp`, `original_experts`) are dropped, and every large weight is
     covered by a *sharded* (non-replicated) spec, so device holds a shard, not a
     full replica;
  3. bounded host RSS                -- process RSS returns to ~baseline after each
     block (no per-layer host accumulation).

It also reports the computed *expected* per-device weight footprint (shapes /
shard factor) so it can be compared against external device telemetry.

NOTE ON DEVICE BYTES: the TT PJRT plugin returns UNIMPLEMENTED for
``xm.get_memory_info`` and there is no tt-smi/pyluwen here, so this test cannot
read actual device DRAM. It validates what our Python code ships (sharded, once,
golden dropped) and the host side exactly; true device residency (e.g. a bf16
shadow kept alongside a bf8 const-eval copy) needs device telemetry.

Runs NO prefill/decode, so it is fast and does not hit the whole-model prefill OOM.

    pytest -svv tests/torch/models/deepseek_v3_1/test_deepseek_v3_1_streaming_single_layer_mem.py
"""

from __future__ import annotations

import gc
import os

import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger

from . import weight_loader
from .test_deepseek_v3_1_e2e_streaming import (
    WEIGHT_DTYPE,
    _alias_rotary_across_layers,
    _block_shard_spec,
    _init_mla_cache,
    _log_mem,
    _make_mesh,
    _malloc_trim,
    _materialize_rotary_caches,
    _ship_module,
    _strip_cpu_golden,
    _top_level_shard_spec,
    _transfer_and_shard_cache,
)

# 3 dense (0-2) + 1 MoE (3): the smallest stack that exercises the big MoE path.
NUM_LAYERS_PROBE = 4
BATCH_SIZE_PROBE = 128
# After shipping a block, RSS must return to within this of the pre-block baseline.
HOST_CLEAR_TOL_GB = 1.5


def _build_skeleton_local(loader):
    from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.modified_modeling_deepseek import (
        DeepseekV3ForCausalLM,
    )

    config = loader._load_config(num_layers=NUM_LAYERS_PROBE)
    config._experts_implementation = "batched_mm"
    config.layer_types = ["full_attention"] * NUM_LAYERS_PROBE
    with torch.device("meta"):
        model = DeepseekV3ForCausalLM(config).eval()
    return model, config


def _rss_gb() -> float:
    import psutil

    _malloc_trim()
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def _cpu_tensor_bytes(module) -> int:
    """Total bytes of CPU-resident parameters + buffers still held by `module`."""
    total = 0
    for sub in module.modules():
        for p in sub._parameters.values():
            if p is not None and p.device.type == "cpu":
                total += p.numel() * p.element_size()
        for b in sub._buffers.values():
            if b is not None and b.device.type == "cpu":
                total += b.numel() * b.element_size()
    return total


def _shard_factor(spec, mesh_sizes) -> int:
    """Number of devices a tensor with `spec` is split across (1 = replicated)."""
    if spec is None:
        return 1
    f = 1
    for axis in spec:
        if axis is None:
            continue
        if isinstance(axis, tuple):
            for a in axis:
                f *= mesh_sizes[a]
        else:
            f *= mesh_sizes[axis]
    return f


def _per_device_weight_bytes(block, mesh_sizes):
    """(expected per-device bytes, replicated-big-tensor list) for `block` given
    its shard spec. Params absent from the spec count as replicated (factor 1)."""
    spec_by_id = {id(t): s for t, s in _block_shard_spec(block).items()}
    per_dev = 0
    replicated_big = []
    for sub in block.modules():
        for name, p in sub._parameters.items():
            if p is None:
                continue
            spec = spec_by_id.get(id(p))
            factor = _shard_factor(spec, mesh_sizes)
            full = p.numel() * p.element_size()
            per_dev += full // factor
            # Flag any >64 MB tensor that is NOT sharded (would waste device DRAM).
            if factor == 1 and full > 64 * 1024 * 1024:
                replicated_big.append((name, full / 1e9))
    return per_dev, replicated_big


@pytest.mark.nightly
@pytest.mark.galaxy
@torch.inference_mode()
def test_streaming_single_layer_mem() -> None:
    from tt_torch.sparse_mlp import A2aSparseMLPWithSharedExperts, enable_sparse_mlp

    from third_party.tt_forge_models.deepseek.deepseek_v3_1.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    from .test_deepseek_v3_1_e2e_streaming import _setup_logging

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
            "experimental_weight_dtype": WEIGHT_DTYPE,
        }
    )

    device = torch_xla.device()
    loader = ModelLoader(
        variant=ModelVariant.DEEPSEEK_V3_1_MODIFIED, num_layers=NUM_LAYERS_PROBE
    )
    mesh, mesh_shape = _make_mesh(loader)
    _, mesh_names = loader.get_mesh_config(xr.global_runtime_device_count())
    mesh_sizes = {name: size for name, size in zip(mesh_names, mesh_shape)}
    max_cache_len = 32

    # ---- skeleton + top-level + cache (all cheap) ----
    model, config = _build_skeleton_local(loader)
    layers = model.model.layers
    _alias_rotary_across_layers(model)
    _materialize_rotary_caches(model, seq_len=max_cache_len, dtype=torch.bfloat16)

    top_sd = weight_loader.load_top_level_state_dict(loader._BF16_WEIGHTS_REPO)
    model.load_state_dict(top_sd, strict=False, assign=True)
    del top_sd
    gc.collect()
    top_spec_by_id = {id(t): s for t, s in _top_level_shard_spec(model).items()}
    _ship_module(model.model.embed_tokens, top_spec_by_id, mesh, device)
    _ship_module(model.model.norm, top_spec_by_id, mesh, device)
    _ship_module(model.lm_head, top_spec_by_id, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()

    cache = _init_mla_cache(config, BATCH_SIZE_PROBE, max_cache_len)
    _transfer_and_shard_cache(cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()

    # ---- per-block flush plumbing (q_len=1) ----
    dummy_hidden = torch.zeros(
        BATCH_SIZE_PROBE, 1, config.hidden_size, dtype=torch.bfloat16
    ).to(device)
    xs.mark_sharding(dummy_hidden, mesh, ("batch", None, None))
    cache_pos_cpu = torch.tensor([0], dtype=torch.long)
    dummy_mask = (
        cache.layers[0]
        .build_causal_mask(cache_pos_cpu, BATCH_SIZE_PROBE, torch.bfloat16, torch.device("cpu"))
        .to(device)
    )
    xs.mark_sharding(dummy_mask, mesh, ("batch", None, None, None))
    dummy_pos_ids = torch.tensor([[0]], dtype=torch.long).to(device)
    dummy_cache_pos = cache_pos_cpu.to(device)
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

    # ---- per-block streaming with validation ----
    results = []
    failures = []
    for i in range(NUM_LAYERS_PROBE):
        block = layers[i]
        baseline_rss = _rss_gb()

        block_sd = weight_loader.load_block_state_dict(loader._BF16_WEIGHTS_REPO, i)
        block.load_state_dict(block_sd, strict=False, assign=True)
        del block_sd
        gc.collect()
        loaded_rss = _rss_gb()  # host footprint of this block's weights

        enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=config)
        _strip_cpu_golden(block)
        gc.collect()

        is_moe = isinstance(block.mlp, A2aSparseMLPWithSharedExperts)
        per_dev_bytes, replicated_big = _per_device_weight_bytes(block, mesh_sizes)

        _ship_module(block, {id(t): s for t, s in _block_shard_spec(block).items()}, mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()
        if do_flush:
            run_block_flush(
                block, dummy_hidden, dummy_mask, dummy_pos_ids, cache, dummy_cache_pos
            )
            torch_xla.sync(wait=True)
            xm.wait_device_ops()
        gc.collect()
        shipped_rss = _rss_gb()

        # ---- checks ----
        cpu_left = _cpu_tensor_bytes(block)
        inner = getattr(block.mlp, "mlp", None)
        golden_mlp = getattr(inner, "_original_mlp", None) if inner is not None else None
        golden_experts = (
            "original_experts" in getattr(inner.experts, "_modules", {})
            if inner is not None and getattr(inner, "experts", None) is not None
            else False
        )
        host_over_baseline = shipped_rss - baseline_rss

        checks = {
            "host_cleared (RSS back to baseline)": host_over_baseline <= HOST_CLEAR_TOL_GB,
            "no CPU tensors left in block": cpu_left == 0,
            "no big replicated weights": len(replicated_big) == 0,
        }
        if is_moe:
            checks["golden _original_mlp dropped"] = golden_mlp is None
            checks["golden original_experts dropped"] = not golden_experts

        ok = all(checks.values())
        if not ok:
            failures.append((i, {k: v for k, v in checks.items() if not v}, replicated_big))

        results.append(
            f"  l{i} ({'MoE ' if is_moe else 'dense'}): "
            f"host_load=+{loaded_rss - baseline_rss:5.1f}GB  "
            f"host_after_ship=+{host_over_baseline:4.1f}GB  "
            f"cpu_left={cpu_left/1e9:.2f}GB  "
            f"expected_dev_weights={per_dev_bytes/1e9:.2f}GB/chip  "
            f"{'PASS' if ok else 'FAIL ' + str([k for k,v in checks.items() if not v])}"
        )
        _log_mem(f"l{i} validated")

    logger.info(
        "[single-layer mem validation] (NO prefill; device bytes not directly "
        "readable on this plugin -- 'expected_dev_weights' is computed from shapes/"
        "sharding)\n" + "\n".join(results)
    )
    for li, failed, repl in failures:
        logger.info(f"[FAIL l{li}] {failed}  replicated_big={repl}")

    assert not failures, (
        f"Streaming infra validation failed for layers {[f[0] for f in failures]}: "
        f"{[f[1] for f in failures]}"
    )
