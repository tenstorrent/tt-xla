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

After every layer is device-resident the whole model is compiled once and then
autoregressively decodes ``NUM_DECODE_TOKENS`` tokens from a single seed token:
each step's argmax'd logit is fed back in as the next step's input.

Like ``kimi_k2_device_test.py`` this is a device/streaming smoke test: it runs
on hardware and prints argmaxed tokens, it does not compare against a CPU
golden. (RoPE caches are left uninitialized by the meta-skeleton build, but
Kimi K2's YaRN rotary recomputes ``inv_freq`` and the cos/sin caches from
config on the first forward, so this does not affect the device run.)

Usage:
    pytest kimi_k2_streaming_test.py -k decode -s
    NUM_LAYERS=61 BATCH_SIZE=64 pytest kimi_k2_streaming_test.py -k decode -s
    NUM_DECODE_TOKENS=32 pytest kimi_k2_streaming_test.py -k decode -s
    NO_DUMMY_FLUSH=1 pytest kimi_k2_streaming_test.py -k decode -s   # debug
"""

import ctypes
import gc
import os
import time

import numpy as np
import pytest
import setproctitle
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from loguru import logger
from torch_xla.distributed.spmd import Mesh

# setproctitle.setproctitle("kimi-stream")


# ============== CONFIGURATION ==============
# Configure via environment variables:
#   NUM_LAYERS=61 BATCH_SIZE=64 pytest kimi_k2_streaming_test.py -k decode -s

DEFAULT_NUM_LAYERS = 4
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_CACHE_LEN = 128
# Number of autoregressive decode steps to run from the single seed token.
# Each step feeds the previous step's argmax'd token back in as the next input.
DEFAULT_NUM_DECODE_TOKENS = 16

# Set to a word to feed the SAME single decode token to EVERY batch row, like
# kimi_k2_device_test.py (which decodes "Hello" across the whole batch). This
# makes the streaming output directly comparable to the device test: with
# identical inputs both should argmax the same next token per row. Set to None
# to use distinct per-row words from DECODE_WORDS instead (good for spotting
# degenerate/identical output, bad for a 1:1 device-test comparison).
UNIFORM_DECODE_WORD = "None"

# Pool of seed words for the decode step. Each batch row gets one word (cycled
# to fill the batch) so the single decode token differs across the batch
# instead of every row decoding the same token. Only used when
# UNIFORM_DECODE_WORD is None.
DECODE_WORDS = [
    "The",
    "Once",
    "Today",
    "Water",
    "Music",
    "Science",
    "History",
    "Mountain",
    "Ocean",
    "Future",
    "Light",
    "Time",
    "Dream",
    "Robot",
    "Hello",
    "Garden",
]


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


@pytest.fixture
def num_decode_tokens():
    return int(os.environ.get("NUM_DECODE_TOKENS", DEFAULT_NUM_DECODE_TOKENS))


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
        logger.info(f"[mem] [{tag:28s}] rss={rss:6.2f} sys={sys_used:6.2f} GB")
    except Exception:
        logger.info(f"[mem] [{tag:28s}] (psutil unavailable)")


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


def materialize_rotary_caches(model, seq_len: int, dtype=torch.bfloat16):
    """Eagerly recompute every rotary embedding's cos/sin cache on CPU.

    The skeleton is built on ``meta`` then ``to_empty``, so each
    ``DeepseekV3*RotaryEmbedding``'s non-persistent ``inv_freq`` /
    ``cos_cached`` / ``sin_cached`` buffers hold uninitialized garbage. They
    are registered ``persistent=False``, so they are absent from the module's
    ``state_dict`` and ``load_state_dict`` never fills them -- even though the
    unsloth checkpoint does ship ``inv_freq`` (it surfaces as an *unexpected*
    key and is filtered out by ``weight_loader._is_loadable_key``), and the
    cos/sin caches are derived and never in the checkpoint at all. Normally
    ``forward`` lazily recomputes them on the first call (its
    ``max_seq_len_cached`` starts ``None``), but in streaming that first call
    is a *per-layer* compiled dummy flush whose recomputed device buffer is a
    transient of that graph and is not reliably retained into the separate
    whole-model decode graph (host-input reclamation can reclaim it). Decode
    then reads stale/garbage rotary tables, attention saturates, and every row
    argmaxes to the same junk token regardless of input.

    Recomputing here (YaRN rebuilds ``inv_freq`` from config scalars, so the
    garbage buffer is irrelevant) and pinning ``max_seq_len_cached`` to
    ``seq_len`` means neither the flush nor the decode recomputes; the correct
    caches are shipped to device as ordinary held buffers alongside the
    weights. ``seq_len`` must be the runtime rotary length, i.e.
    ``cache.get_max_cache_shape()`` == ``max_cache_len``."""
    count = 0
    for module in model.modules():
        if hasattr(module, "_set_cos_sin_cache") and hasattr(module, "inv_freq"):
            module._set_cos_sin_cache(seq_len=seq_len, device="cpu", dtype=dtype)
            count += 1
    logger.info(
        f"Materialized rotary cos/sin caches on {count} module(s) (seq_len={seq_len})"
    )
    # Guard against a silent no-op: if the rotary class is ever refactored and
    # _set_cos_sin_cache disappears/renames, count drops to 0, the caches stay
    # garbage, and the degenerate-output bug returns quietly. Fail loudly here.
    if count == 0:
        raise RuntimeError(
            "materialize_rotary_caches matched no rotary modules -- expected one "
            "per decoder layer (DeepseekV3*RotaryEmbedding._set_cos_sin_cache). "
            "The rotary embedding API likely changed; update this helper."
        )


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


def reinit_cache_buffers(cache, mesh, device):
    """Replace the MLA cache tensors with FRESH zeroed, sharded device buffers.

    The per-layer dummy flushes do two things to the cache: (1) write garbage
    into it via the attention update, and (2) under host-input reclamation
    (``TT_XLA_DEALLOCATE_HOST_INPUTS_AFTER_MIGRATION``) free the host backing
    of the original cache shards once the first flush migrates them to device.

    Re-zeroing in place (``.zero_()``) re-stages the buffer and makes the next
    program rebuild a multi-device host tensor from those now-freed per-shard
    host copies -> "Tensor is not allocated". Instead we ship brand-new zeroed
    device tensors so the real decode consumes clean, freshly-allocated buffers
    whose host backing is still live. Mirrors the streaming package's
    ``_reinit_mutable_kv_buffers``."""
    kv_spec = ("batch", None, None, None)
    for layer in cache.layers:
        kv_shape = tuple(layer.compressed_kv.shape)
        pe_shape = tuple(layer.k_pe.shape)
        fresh_kv = torch.zeros(kv_shape, dtype=torch.bfloat16).to(device)
        fresh_pe = torch.zeros(pe_shape, dtype=torch.bfloat16).to(device)
        layer.compressed_kv = fresh_kv
        layer.k_pe = fresh_pe
        layer.keys = fresh_kv
        layer.values = fresh_pe
        torch._dynamo.mark_static_address(layer.compressed_kv)
        torch._dynamo.mark_static_address(layer.k_pe)
        xs.mark_sharding(layer.compressed_kv, mesh, kv_spec)
        xs.mark_sharding(layer.k_pe, mesh, kv_spec)


# ============== STREAMING DECODE TEST ==============


def test_kimi_k2_streaming_decode(
    num_layers, batch_size, max_cache_len, num_decode_tokens
):
    """Stream the model onto device one layer at a time, then autoregressively
    decode ``num_decode_tokens`` tokens from a single seed token. Peak host RAM
    stays bounded to ~one layer of weights."""
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
    logger.info(f"Mesh: {mesh.shape()}  num_devices={xr.global_runtime_device_count()}")
    mesh_shape = mesh_shape_for(xr.global_runtime_device_count())

    # ---- 1. Build a weight-less CPU skeleton ----
    logger.info(f"Building Kimi K2 skeleton with {num_layers} layers...")
    loader = ModelLoader(
        variant=ModelVariant.KIMI_K2_BASE_MODIFIED, num_layers=num_layers
    )
    model = loader.build_skeleton()
    config = loader.config
    tokenizer = loader.tokenizer
    layers = model.model.layers
    n_layers = len(layers)
    log_mem("skeleton")

    # ``to_empty`` left the rotary cos/sin caches uninitialized. Recompute them
    # on CPU now (for the fixed rotary length = max_cache_len) so they ship to
    # device as correct, held buffers instead of being lazily (and unreliably)
    # rebuilt inside a per-layer flush graph. See materialize_rotary_caches.
    materialize_rotary_caches(model, seq_len=max_cache_len, dtype=torch.bfloat16)

    # ---- 2. Ship top-level params (embed / norm / head) ----
    logger.info("Shipping top-level params (embed / norm / lm_head)...")
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
    logger.info("Allocating + sharding MLA cache on device...")
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
    # Reuse the cache's own mask builder (the layers are MLAStaticLayer) so the
    # flush mask matches exactly what the model/cache produces at runtime --
    # built on CPU here, then shipped + sharded like every other input.
    cpu_cache_pos = torch.tensor([flush_pos], dtype=torch.long)
    dummy_mask = (
        cache.layers[0]
        .build_causal_mask(
            cache_position=cpu_cache_pos,
            batch_size=batch_size,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        .to(device)
    )
    xs.mark_sharding(dummy_mask, mesh, ("batch", None, None, None))
    dummy_pos_ids = torch.tensor([[flush_pos]], dtype=torch.long).to(device)
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

    # ---- 5. Per-layer: load -> sparse-MLP -> ship -> dummy-flush -> free ----
    logger.info(f"Streaming {n_layers} layers onto device (dummy_flush={do_flush})...")
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

    # ---- 6. All layers device-resident: re-init KV the flushes dirtied ----
    # Ship FRESH device buffers rather than zeroing in place, so the decode
    # does not reuse cache shards whose host backing the flushes already freed.
    logger.info("All layers resident. Re-initializing MLA cache for the real decode...")
    reinit_cache_buffers(cache, mesh, device)
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log_mem("post-stream")

    # ---- 7. Whole-model compile + autoregressive multi-token decode ----
    logger.info("Compiling whole model...")
    compiled_model = torch.compile(model, backend="tt")

    # Either the same token for every row (UNIFORM_DECODE_WORD, comparable to
    # the device test) or a distinct word per row cycled from DECODE_WORDS.
    if UNIFORM_DECODE_WORD is not None:
        seed_words = [UNIFORM_DECODE_WORD] * batch_size
    else:
        seed_words = [DECODE_WORDS[i % len(DECODE_WORDS)] for i in range(batch_size)]
    seed_token_ids = [
        tokenizer.encode(word, add_special_tokens=False)[0] for word in seed_words
    ]
    input_ids = (
        torch.tensor(seed_token_ids, dtype=torch.long).unsqueeze(1).to(device)
    )  # (batch_size, 1)
    xs.mark_sharding(input_ids, mesh, ("batch", None))
    cache_position = torch.tensor([0], dtype=torch.long).to(device)

    # Autoregressive loop: each step feeds the previous step's argmax'd token
    # back in as the single decode input, advancing cache_position by one. We
    # accumulate the per-step predicted ids on host so we can print each row's
    # full generated continuation at the end.
    logger.info(f"Running {num_decode_tokens} autoregressive decode step(s)...")
    generated_ids = [[] for _ in range(batch_size)]
    # Per-step wall-clock for the full-model decode. The .cpu() transfer below
    # blocks until the device finishes, so timing the forward + materialization
    # captures the full per-step decode latency. Step 0 also pays the one-time
    # torch.compile cost, so we report it separately from the steady state.
    step_times = []
    with torch.no_grad():
        for step in range(num_decode_tokens):
            t0 = time.perf_counter()
            output = compiled_model(
                input_ids=input_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            logits = output.logits.cpu()
            step_times.append(time.perf_counter() - t0)
            if step == 0:
                # Expected (batch_size, seq_len=1, vocab). Under SPMD the
                # gathered host tensor's batch dim may be a single shard (e.g.
                # batch_size // batch_axis) rather than the full batch -- log it
                # so a partial gather is obvious.
                logger.info(
                    f"[STREAMING DECODE] logits shape: {tuple(logits.shape)}"
                )

            # (N,) one predicted token id per row for this step.
            predicted_ids = logits[:, -1].argmax(dim=-1)
            for i in range(predicted_ids.shape[0]):
                generated_ids[i].append(int(predicted_ids[i]))

            # Feed the argmax'd token back in as the next step's input and
            # advance the cache position by one (mirrors kimi_k2_device_test's
            # autoregressive update).
            next_token = predicted_ids.unsqueeze(1).to(device)  # (N, 1)
            xs.mark_sharding(next_token, mesh, ("batch", None))
            input_ids = next_token
            cache_position = cache_position[-1:] + 1

    # Per-step full-model decode timing. Step 0 includes the one-time
    # torch.compile, so report it on its own and compute mean/min/max/var over
    # the steady-state steps (1..N-1) when there is more than one step.
    times = np.asarray(step_times, dtype=np.float64)
    steady = times[1:] if times.size > 1 else times
    stat_lines = [
        "[STREAMING DECODE] full-model decode time per step (s):",
        f"  all steps: {', '.join(f'{t:.4f}' for t in times)}",
        f"  step 0 (incl. compile): {times[0]:.4f}",
        (
            f"  steady-state (steps 1..{times.size - 1}, n={steady.size}): "
            f"mean={steady.mean():.4f}  min={steady.min():.4f}  "
            f"max={steady.max():.4f}  var={steady.var():.6f}"
        ),
    ]
    logger.info("\n".join(stat_lines))

    # Decode each row's full generated continuation. batch_decode treats each
    # inner list as its own token sequence, so we get one string per row.
    decoded = tokenizer.batch_decode(generated_ids)
    # Build the whole report as ONE string and log it in a single call. Under
    # SPMD multiple rank processes share stdout; one logger.info per row gets
    # interleaved/clobbered (which is why early rows looked "missing"). A single
    # atomic write keeps every row intact.
    lines = [
        f"[STREAMING DECODE] {len(decoded)} row(s); "
        f"{num_decode_tokens} token(s) generated per row:"
    ]
    for i in range(len(decoded)):
        seed = seed_words[i] if i < len(seed_words) else "?"
        lines.append(f"  User {i}: {seed!r} -> {decoded[i]!r}")
    logger.info("\n".join(lines))
