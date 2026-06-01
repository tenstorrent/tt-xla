# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2 streaming decode test.

Runs a single decode step on TT device, but loads the model weights one
transformer layer at a time instead of all at once. This bounds peak host
(CPU) RAM to roughly a single layer's worth of weights, so a checkpoint whose
full weight set does not fit on host can still be loaded and run.

This is the Kimi-K2 counterpart of the ``streaming/`` package described in
``STREAMING_DISTILLED.md``, distilled into a single self-contained file for the
decode-only case. The per-layer lifecycle is:

    load layer weights (CPU)  ->  enable sparse MLP  ->  ship to device (lazy)
    ->  dummy-flush (force the lazy host->device transfer)  ->  free CPU storage

After every layer is device-resident the whole model is compiled once and a
single decode step is run.

Like ``kimi_k2_device_test.py`` this is a device/streaming smoke test: it runs
on hardware and prints argmaxed tokens, it does not compare against a CPU
golden. (RoPE caches are left uninitialized by the meta-skeleton build, but
Kimi K2's YaRN rotary recomputes ``inv_freq`` and the cos/sin caches from
config on the first forward, so this does not affect the device run.)

Usage:
    pytest kimi_k2_streaming_test.py -k decode -s
    NUM_LAYERS=61 BATCH_SIZE=64 pytest kimi_k2_streaming_test.py -k decode -s
    NO_DUMMY_FLUSH=1 pytest kimi_k2_streaming_test.py -k decode -s   # debug
"""

import ctypes
import gc
import os

import numpy as np
import pytest
import setproctitle
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

setproctitle.setproctitle("kimi-stream")


# ============== CONFIGURATION ==============
# Configure via environment variables:
#   NUM_LAYERS=61 BATCH_SIZE=64 pytest kimi_k2_streaming_test.py -k decode -s

DEFAULT_NUM_LAYERS = 4
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CACHE_LEN = 128


# ============== PYTEST FIXTURES ==============


@pytest.fixture
def num_layers():
    return int(os.environ.get("NUM_LAYERS", DEFAULT_NUM_LAYERS))


@pytest.fixture
def batch_size():
    return int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))


@pytest.fixture
def max_cache_len():
    return int(os.environ.get("MAX_CACHE_LEN", DEFAULT_MAX_CACHE_LEN))


# ============== MEMORY / MESH HELPERS ==============


def _malloc_trim():
    """Force glibc to return freed arenas to the OS so RSS tracks live
    allocations. Without this, freed per-layer CPU storage stays in the
    process and the streaming bound is impossible to observe."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log_mem(tag: str):
    """Trim, then log resident-set / system memory. Best-effort (psutil)."""
    _malloc_trim()
    try:
        import psutil

        p = psutil.Process(os.getpid())
        rss = p.memory_info().rss / 1e9
        sys_used = psutil.virtual_memory().used / 1e9
        print(f"[mem] [{tag:28s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")
    except Exception:
        print(f"[mem] [{tag:28s}] (psutil unavailable)")


def setup_spmd():
    """Initialize SPMD mode for multi-device."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def mesh_shape_for(num_devices: int):
    """Mesh shape supporting 64/32/8 devices (matches kimi_k2_device_test)."""
    if num_devices == 64:
        return (4, 16)
    elif num_devices == 32:  # Galaxy
        return (4, 8)
    elif num_devices == 8:  # llmbox
        return (2, 4)
    raise ValueError(f"Kimi K2: unsupported num_devices={num_devices}")


def create_mesh() -> Mesh:
    """Create device mesh with ('batch', 'model') axis names."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape = mesh_shape_for(num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))


# ============== DEVICE SHIP HELPERS ==============


def upload_with_sharding(cpu_tensor, mesh, partition_spec, device):
    """Move a CPU tensor to device (lazy) and annotate sharding.
    ``partition_spec=None`` leaves the tensor replicated."""
    xla_t = cpu_tensor.to(device)
    if partition_spec is not None:
        xs.mark_sharding(xla_t, mesh, partition_spec)
    return xla_t


def ship_module_to_device(module, spec_by_id, mesh, device):
    """Replace every CPU param/buffer in ``module`` with a device handle,
    dropping the CPU storage. ``spec_by_id`` maps ``id(cpu_tensor)`` to a
    partition spec; tensors not in the map upload replicated.

    The ``.to(device)`` transfer is LAZY -- it only actually executes when a
    computation that consumes these tensors runs (the dummy flush)."""
    for sub in module.modules():
        for name, p in list(sub._parameters.items()):
            if p is None or p.device.type != "cpu":
                continue
            spec = spec_by_id.get(id(p))
            xla_t = upload_with_sharding(p.data.detach(), mesh, spec, device)
            sub._parameters[name] = torch.nn.Parameter(xla_t, requires_grad=False)
        for name, b in list(sub._buffers.items()):
            if b is None or b.device.type != "cpu":
                continue
            spec = spec_by_id.get(id(b))
            xla_t = upload_with_sharding(b.detach(), mesh, spec, device)
            sub._buffers[name] = xla_t


# ============== MLA CACHE HELPERS ==============


def init_mla_cache(config, batch_size: int, max_cache_len: int):
    """Create + lazily allocate an MLA cache on CPU."""
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


def transfer_and_shard_cache(cache, mesh, device):
    """Move the MLA cache tensors to device and batch-shard them."""
    kv_spec = ("batch", None, None, None)
    for layer in cache.layers:
        layer.compressed_kv = layer.compressed_kv.to(device)
        layer.k_pe = layer.k_pe.to(device)
        layer.keys = layer.compressed_kv
        layer.values = layer.k_pe
        torch._dynamo.mark_static_address(layer.compressed_kv)
        torch._dynamo.mark_static_address(layer.k_pe)
        xs.mark_sharding(layer.compressed_kv, mesh, kv_spec)
        xs.mark_sharding(layer.k_pe, mesh, kv_spec)


def zero_cache(cache):
    """Re-zero the MLA cache in place (the per-layer dummy flushes write
    garbage into it; clear before the real decode)."""
    for layer in cache.layers:
        layer.compressed_kv.zero_()
        layer.k_pe.zero_()


def build_decode_mask(batch_size: int, max_cache_len: int, cache_pos: int, dtype):
    """4D causal mask (batch, 1, 1, max_cache_len) for a single decode token
    at position ``cache_pos``. Mirrors MLAStaticLayer.build_causal_mask."""
    key_idx = torch.arange(max_cache_len)
    pos = torch.tensor([cache_pos])
    causal = (key_idx.unsqueeze(0) > pos.unsqueeze(1)).to(dtype)  # (1, max_cache_len)
    fill = torch.finfo(dtype).min
    return (causal * fill).unsqueeze(0).unsqueeze(0).expand(batch_size, 1, 1, -1)


# ============== STREAMING DECODE TEST ==============


def test_kimi_k2_streaming_decode(num_layers, batch_size, max_cache_len):
    """Stream the model onto device one layer at a time, then run one decode
    step. Peak host RAM stays bounded to ~one layer of weights."""
    from tt_torch.sparse_mlp import enable_sparse_mlp

    from third_party.tt_forge_models.kimi_k2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    do_flush = os.environ.get("NO_DUMMY_FLUSH", "0") != "1"

    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()

    # One dynamo cache entry per unique layer shape during the per-layer flush
    # (plus the whole-model prefill/decode). Default 8 is far below a deep
    # transformer stack.
    torch._dynamo.config.cache_size_limit = 1000

    # The critical streaming knob: keep const-eval inputs (the weights) in
    # DEVICE DRAM instead of bouncing them back to host system memory, which
    # would silently repopulate the host shadow we just freed and break the
    # bound. Must be set AFTER the TT ComputationClient is up (set_device_type).
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": 0,  # Minimal optimization for stability
            "enable_trace": False,  # Disabled due to topk indices issue
            "experimental_weight_dtype": "bfp_bf8",
            "enable_const_eval_inputs_to_system_memory": False,
        }
    )

    mesh = create_mesh()
    print(f"\nMesh: {mesh.shape()}  num_devices={xr.global_runtime_device_count()}")
    mesh_shape = mesh_shape_for(xr.global_runtime_device_count())

    # ---- 1. Build a weight-less CPU skeleton ----
    print(f"\nBuilding Kimi K2 skeleton with {num_layers} layers...")
    loader = ModelLoader(
        variant=ModelVariant.KIMI_K2_INSTRUCT_MODIFIED, num_layers=num_layers
    )
    model = loader.build_skeleton()
    config = loader.config
    tokenizer = loader.tokenizer
    layers = model.model.layers
    n_layers = len(layers)
    log_mem("skeleton")

    # ---- 2. Ship top-level params (embed / norm / head) ----
    print("Shipping top-level params (embed / norm / lm_head)...")
    top_sd = loader.load_top_level_state_dict()
    missing, unexpected = model.load_state_dict(top_sd, strict=False)
    if unexpected:
        raise RuntimeError(f"top-level load: unexpected keys {sorted(unexpected)[:8]}")
    del top_sd
    gc.collect()
    top_spec = loader.load_top_level_shard_spec(model)
    top_spec_by_id = {id(t): s for t, s in top_spec.items()}
    ship_module_to_device(model.model.embed_tokens, top_spec_by_id, mesh, device)
    ship_module_to_device(model.model.norm, top_spec_by_id, mesh, device)
    ship_module_to_device(model.lm_head, top_spec_by_id, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    gc.collect()
    log_mem("post-top-level")

    # ---- 3. Pre-allocate + ship the MLA cache (persistent, on device) ----
    # The cache must exist on device before the per-layer dummy flush, which
    # writes into cache.layers[i] via the attention update.
    print("Allocating + sharding MLA cache on device...")
    cache = init_mla_cache(config, batch_size, max_cache_len)
    transfer_and_shard_cache(cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log_mem("post-cache")

    # ---- 4. Dummy decode inputs for the per-layer flush (q_len = 1) ----
    flush_pos = 0
    dummy_hidden = torch.zeros(
        batch_size, 1, config.hidden_size, dtype=torch.bfloat16
    ).to(device)
    xs.mark_sharding(dummy_hidden, mesh, ("batch", None, None))
    dummy_mask = build_decode_mask(
        batch_size, max_cache_len, flush_pos, torch.bfloat16
    ).to(device)
    xs.mark_sharding(dummy_mask, mesh, ("batch", None, None, None))
    dummy_pos_ids = torch.tensor([[flush_pos]], dtype=torch.long).to(device)
    dummy_cache_pos = torch.tensor([flush_pos], dtype=torch.long).to(device)
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

    # ---- 5. Per-layer: load -> sparse-MLP -> ship -> dummy-flush -> free ----
    print(f"\nStreaming {n_layers} layers onto device (dummy_flush={do_flush})...")
    for i in range(n_layers):
        block = layers[i]

        # 5a. Load this layer's weights into the skeleton's bf16 storage.
        block_sd = loader.load_block_state_dict(i)
        _, unexpected = block.load_state_dict(block_sd, strict=False)
        if unexpected:
            raise RuntimeError(
                f"layer {i} load: unexpected keys {sorted(unexpected)[:8]}"
            )
        del block_sd
        gc.collect()

        # 5b. Swap MoE -> sparse all-to-all MLP (no-op for the dense layer 0).
        #     enable_sparse_mlp stacks the expert weights and (by default)
        #     frees the per-expert CPU storage.
        enable_sparse_mlp(block, mesh=mesh_shape, cluster_axis=0, config=config)
        gc.collect()

        # 5c. Ship the layer's params to device (lazy) with sharding.
        block_spec = loader.load_block_shard_spec(block)
        block_spec_by_id = {id(t): s for t, s in block_spec.items()}
        ship_module_to_device(block, block_spec_by_id, mesh, device)
        torch_xla.sync(wait=True)
        xm.wait_device_ops()
        gc.collect()

        # 5d. Dummy flush: force the lazy host->device transfer to execute so
        #     the plugin's host staging buffers (and the CPU weight storage)
        #     can actually be released. Mandatory for the host-RAM bound.
        if do_flush:
            _ = run_block_flush(
                block,
                dummy_hidden,
                dummy_mask,
                dummy_pos_ids,
                cache,
                dummy_cache_pos,
            )
            torch_xla.sync(wait=True)
            xm.wait_device_ops()

        gc.collect()
        _malloc_trim()
        log_mem(f"layer {i:>3}")

    # ---- 6. All layers device-resident: re-zero KV the flushes dirtied ----
    print("\nAll layers resident. Re-zeroing MLA cache for the real decode...")
    zero_cache(cache)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log_mem("post-stream")

    # ---- 7. Whole-model compile + single decode step ----
    print("Compiling whole model...")
    compiled_model = torch.compile(model, backend="tt")

    single_token = tokenizer.encode("Hello", return_tensors="pt")[:, :1]
    input_ids = single_token.expand(batch_size, -1).contiguous().to(device)
    xs.mark_sharding(input_ids, mesh, ("batch", None))
    cache_position = torch.tensor([0], dtype=torch.long).to(device)

    print("Running decode step...")
    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )

    logits = output.logits.cpu()
    predicted_ids = logits[:, -1].argmax(dim=-1)
    decoded = tokenizer.batch_decode(predicted_ids)
    print("\n[STREAMING DECODE] Predicted tokens:")
    for i, text in enumerate(decoded[: min(8, batch_size)]):
        print(f"  User {i}: {repr(text)}")
