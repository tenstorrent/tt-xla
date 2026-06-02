# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Kimi K2 CPU-only cold-decode reference test.

Runs a single "cold" decode step on plain CPU torch (no torch_xla, no SPMD, no
sharding, no sparse MLP) so its argmaxed tokens can be compared against the TT
device / streaming tests for output parity. "Cold" means the same scenario the
device smoke tests use:

    * a freshly ZEROED MLA cache (no prefill), and
    * a single decode token at cache_position 0.

This is the canonical / golden path: the MoE runs in its DENSE form
(``DeepseekV3MoE.forward``), which is the unsharded reference the device's
all-to-all sparse MLP is supposed to reproduce. So if the device output diverges
from this, the bug is on the device/sharding side, not the math.

No streaming and no memory optimization -- the whole (few-layer) model is loaded
into host RAM at once. Intended for small depths (NUM_LAYERS=2 or 4).

Because every batch row is fed the SAME token (UNIFORM_DECODE_WORD), every row
is identical, so BATCH_SIZE does not change the predicted token -- it only needs
to match the device test's batch if you want identical shapes. A small batch is
plenty for the reference.

Usage:
    pytest kimi_k2_cpu_test.py -k cold_decode -s
    NUM_LAYERS=4 BATCH_SIZE=8 pytest kimi_k2_cpu_test.py -k cold_decode -s
    DTYPE=float32 pytest kimi_k2_cpu_test.py -k cold_decode -s   # fp32 golden
"""

import gc
import importlib.util
import os
import sys
import types

import pytest
import torch
from loguru import logger

# ============== CONFIGURATION ==============
# Configure via environment variables:
#   NUM_LAYERS=4 BATCH_SIZE=8 pytest kimi_k2_cpu_test.py -k cold_decode -s

DEFAULT_NUM_LAYERS = 2
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_CACHE_LEN = 128

# Single token fed to every batch row, matching kimi_k2_device_test.py (decode)
# and kimi_k2_streaming_test.py's UNIFORM_DECODE_WORD, so the three tests are
# directly comparable.
UNIFORM_DECODE_WORD = "Hello"

# bfloat16 matches the device run (same rounding) and is the right comparison
# target. Set DTYPE=float32 for a higher-precision golden reference.
_DTYPE_BY_NAME = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


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
def dtype():
    return _DTYPE_BY_NAME[os.environ.get("DTYPE", "bfloat16").lower()]


# ============== DEPENDENCY LOADING (no torch_xla) ==============
# This is a CPU-only test, but importing the kimi package normally
# (`third_party.tt_forge_models.kimi_k2...`) runs its __init__ chain, which
# imports `loader.py` -> `torch_xla` and `tt_torch` (the device backend). On a
# stale/rebuilding pjrt_plugin_tt.so that import crashes. We don't need any of
# that here, so we load ONLY the three pure-torch modules we use
# (configuration_deepseek, modified_modeling_deepseek, weight_loader) plus the
# MLA cache, directly from their files via importlib -- bypassing the package
# __init__ and loader.py entirely.

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
_KIMI_PKG_DIR = os.path.join(
    _REPO_ROOT, "third_party", "tt_forge_models", "kimi_k2", "pytorch"
)
_MLA_CACHE_PATH = os.path.join(
    _REPO_ROOT, "tests", "infra", "utilities", "torch_mla_cache.py"
)
_SYNTH_PKG = "_kimi_cpu_pkg"


def _load_file_module(name: str, path: str):
    """exec a single .py file as module ``name`` (registered in sys.modules)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_kimi_deps():
    """Load the pure-torch kimi modules + MLA cache without importing the
    package (which would pull torch_xla/tt_torch). Returns a namespace with
    DeepseekV3Config, DeepseekV3ForCausalLM, weight_loader module, MLACache."""
    # Synthetic parent package so the modules' relative import
    # (`from .configuration_deepseek import ...`) resolves against _KIMI_PKG_DIR.
    if _SYNTH_PKG not in sys.modules:
        pkg = types.ModuleType(_SYNTH_PKG)
        pkg.__path__ = [_KIMI_PKG_DIR]
        pkg.__package__ = _SYNTH_PKG
        sys.modules[_SYNTH_PKG] = pkg

    cfg_mod = _load_file_module(
        f"{_SYNTH_PKG}.configuration_deepseek",
        os.path.join(_KIMI_PKG_DIR, "configuration_deepseek.py"),
    )
    modeling_mod = _load_file_module(
        f"{_SYNTH_PKG}.modified_modeling_deepseek",
        os.path.join(_KIMI_PKG_DIR, "modified_modeling_deepseek.py"),
    )
    wl_mod = _load_file_module(
        f"{_SYNTH_PKG}.weight_loader",
        os.path.join(_KIMI_PKG_DIR, "weight_loader.py"),
    )
    mla_mod = _load_file_module("_kimi_cpu_mla_cache", _MLA_CACHE_PATH)

    return types.SimpleNamespace(
        DeepseekV3Config=cfg_mod.DeepseekV3Config,
        DeepseekV3ForCausalLM=modeling_mod.DeepseekV3ForCausalLM,
        weight_loader=wl_mod,
        MLACache=mla_mod.MLACache,
        config_json=os.path.join(_KIMI_PKG_DIR, "config.json"),
    )


# ============== MODEL / CACHE HELPERS ==============


def ensure_checkpoint_shards(num_layers: int, repo_id: str):
    """Pre-download ONLY the safetensors shards needed for the requested layers.

    The tt-forge-models kimi weight_loader is index-driven: it reads
    ``model.safetensors.index.json`` and only opens the shard files that contain
    the requested prefixes (embed/norm/lm_head + ``model.layers.{i}.``), reading
    them from ``KIMI_K2_CHECKPOINT_DIR``. It does NOT itself download. So here we
    replicate that same shard selection, fetch just those files from the unsloth
    BF16 HF repo via ``hf_hub_download`` (cached + idempotent on re-runs), and
    point ``KIMI_K2_CHECKPOINT_DIR`` at the resulting snapshot directory.

    If ``KIMI_K2_CHECKPOINT_DIR`` is already set (e.g. a local mount), it is
    respected and nothing is downloaded."""
    if os.environ.get("KIMI_K2_CHECKPOINT_DIR"):
        logger.info(
            f"[cpu] KIMI_K2_CHECKPOINT_DIR already set to "
            f"{os.environ['KIMI_K2_CHECKPOINT_DIR']}; skipping HF download"
        )
        return

    import json

    from huggingface_hub import hf_hub_download

    index_path = hf_hub_download(repo_id, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # Same prefixes the weight_loader resolves for a layer subset + top-level.
    prefixes = tuple(
        ["model.embed_tokens.", "model.norm.", "lm_head."]
        + [f"model.layers.{i}." for i in range(num_layers)]
    )
    shards = sorted({s for k, s in weight_map.items() if k.startswith(prefixes)})
    logger.info(
        f"[cpu] {num_layers} layer(s) -> {len(shards)} shard file(s) from {repo_id}"
    )
    for i, shard in enumerate(shards):
        logger.info(f"[cpu] Fetching shard {i + 1}/{len(shards)}: {shard}")
        hf_hub_download(repo_id, shard)

    snapshot_dir = os.path.dirname(index_path)
    os.environ["KIMI_K2_CHECKPOINT_DIR"] = snapshot_dir
    logger.info(f"[cpu] KIMI_K2_CHECKPOINT_DIR -> {snapshot_dir}")


def load_cpu_model(num_layers: int, dtype: torch.dtype):
    """Load the Kimi K2 model on CPU with all weights and DENSE MoE.

    Mirrors ModelLoader.load_model's construction (meta build -> to_empty ->
    load real checkpoint weights) but deliberately SKIPS ``enable_sparse_mlp``:
    the sparse all-to-all MLP is a device/sharding transform, whereas the CPU
    reference wants the plain dense ``DeepseekV3MoE`` math.

    Non-persistent RoPE buffers are left to be recomputed lazily on the first
    forward (correct under eager CPU, where the recompute happens in-line)."""
    from transformers import AutoTokenizer

    deps = load_kimi_deps()
    weight_loader = deps.weight_loader

    # Config from the package's local config.json (mirrors ModelLoader).
    config = deps.DeepseekV3Config.from_json_file(deps.config_json)
    config.num_hidden_layers = num_layers
    # Tokenizer from the canonical moonshotai repo (the unsloth reupload ships a
    # broken tokenization module) -- mirrors ModelLoader._load_tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(
        "moonshotai/Kimi-K2-Instruct", trust_remote_code=True
    )
    logger.info(
        f"[cpu] Building Kimi K2: num_hidden_layers={config.num_hidden_layers}, "
        f"hidden_size={config.hidden_size}, dtype={dtype}"
    )

    # Construct in bf16 on meta (cheap), then materialize empty CPU storage.
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            model = deps.DeepseekV3ForCausalLM(config)
    finally:
        torch.set_default_dtype(prev_dtype)
    model = model.to_empty(device="cpu").to(dtype)

    # Download only the shards needed for these layers, then load them.
    ensure_checkpoint_shards(config.num_hidden_layers, weight_loader.REPO_ID)
    sd = weight_loader.load_transformer_state_dict(
        list(range(config.num_hidden_layers))
    )
    if dtype != torch.bfloat16:
        sd = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logger.info(
        f"[cpu] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}"
    )
    del sd
    gc.collect()
    return model.eval(), config, tokenizer


def init_zeroed_mla_cache(config, batch_size: int, max_cache_len: int, dtype):
    """Create a freshly ZEROED MLA cache on CPU (no prefill)."""
    MLACache = load_kimi_deps().MLACache

    cache = MLACache(config=config, max_cache_len=max_cache_len)
    text_config = config.get_text_config(decoder=True)
    kv_lora_rank = text_config.kv_lora_rank
    qk_rope_head_dim = text_config.qk_rope_head_dim

    dummy_kv = torch.zeros((batch_size, 1, 1, kv_lora_rank), dtype=dtype)
    dummy_pe = torch.zeros((batch_size, 1, 1, qk_rope_head_dim), dtype=dtype)
    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)
    return cache


# ============== CPU COLD-DECODE TEST ==============


def test_kimi_k2_cpu_cold_decode(num_layers, batch_size, max_cache_len, dtype):
    """Load the model on CPU and run one cold decode step; print argmaxed
    tokens for parity comparison against the device / streaming tests."""
    model, config, tokenizer = load_cpu_model(num_layers, dtype)

    # ---- Cold decode inputs: single token, zeroed cache, position 0 ----
    cache = init_zeroed_mla_cache(config, batch_size, max_cache_len, dtype)

    # Same single token for every row (matches device/streaming defaults).
    token_id = tokenizer.encode(UNIFORM_DECODE_WORD, add_special_tokens=False)[0]
    input_ids = torch.full((batch_size, 1), token_id, dtype=torch.long)
    cache_position = torch.tensor([0], dtype=torch.long)

    logger.info(
        f"[cpu] Running cold decode: word={UNIFORM_DECODE_WORD!r} token_id={token_id} "
        f"batch={batch_size} max_cache_len={max_cache_len}"
    )
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )

    logits = output.logits  # (batch, 1, vocab)
    predicted_ids = logits[:, -1].argmax(dim=-1)  # (batch,)
    decoded = tokenizer.batch_decode(predicted_ids.unsqueeze(-1))

    # All rows are identical (same input token), so report the shared result
    # plus a couple of rows as a sanity check. Single atomic log call.
    lines = [
        f"[CPU COLD DECODE] logits shape: {tuple(logits.shape)}",
        f"[CPU COLD DECODE] {UNIFORM_DECODE_WORD!r} (id={token_id}) -> "
        f"{decoded[0]!r} (id={int(predicted_ids[0])})",
    ]
    for i in range(min(4, len(decoded))):
        lines.append(f"  User {i}: {decoded[i]!r} (id={int(predicted_ids[i])})")
    logger.info("\n".join(lines))

    # Every row got the same token (deterministic, uniform input).
    assert (
        predicted_ids == predicted_ids[0]
    ).all(), "Rows diverged despite identical input -- nondeterminism in the CPU path"
