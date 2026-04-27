# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Debug bfp4 accuracy for GPT-OSS-120B layer 18 via codegen + CPU matmul bypass.

Pipeline
--------
1. Construct a 1-layer GPT-OSS-120B model (config.num_hidden_layers = 1).
   HF's streaming load only materializes layer 0's weights in CPU RAM — layers
   1..35 are NEVER loaded.
2. Overwrite layer 0's parameters with *layer 18's* weights by streaming only
   the safetensor shards that contain "model.layers.18.*". MoE gate_up_proj /
   down_proj are stored as mxfp4 (blocks + scales) and are dequantized on the
   fly via transformers.integrations.mxfp4.convert_moe_packed_tensors.
3. Apply the Galaxy 4x8 SPMD mesh + _moe_throughput_galaxy_shard_spec_fn
   sharding and the test's weight_dtype_overrides (bfp4 for MoE projections).
4. Wrap in ModelWithCache (StaticCache as registered buffers so they follow
   model.to(device)), and call codegen_py() to emit main.py + consteval.py.
5. Post-process with codegen_accuracy.run_alchemist_cpu_bypass() to create
   main_cpu_bypass.py — identical, but with every bfp4 matmul swapped for
   ttnn.typecast(bfp4 -> BFLOAT16) -> ttnn.from_device -> torch.matmul.

Usage
-----
    source venv/activate
    python tests/benchmark/debug_bfp4_layer18.py

Outputs
-------
    tests/benchmark/bfp4_layer18_output_prompt/main.py
    tests/benchmark/bfp4_layer18_output_prompt/consteval.py      (if const-eval emitted)
    tests/benchmark/bfp4_layer18_output_prompt/main_cpu_bypass.py
    tests/benchmark/bfp4_layer18_output_prompt/<tensor serialization files>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "tests" / "benchmark"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARK_DIR))

# Must be set BEFORE any torch_xla import so the Shardy convert pass runs.
os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.cache_utils import StaticCache
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.utils.quantization_config import Mxfp4Config

from benchmarks.codegen_accuracy import run_alchemist_cpu_bypass
from tt_torch.weight_dtype import apply_weight_dtype_overrides

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

MODEL_NAME = "openai/gpt-oss-120b"
TARGET_LAYER = 18
BATCH_SIZE = 64
INPUT_SEQUENCE_LENGTH = 128
EXPORT_PATH = str(BENCHMARK_DIR / "bfp4_layer18_output_prompt")

# Mirrors test_gpt_oss_120b_tp_galaxy_batch_size_64 (test_llms.py:1644-1649).
WEIGHT_DTYPE_OVERRIDES = {
    "model.layers.*.mlp.router.weight": "bf16",
    "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
    "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
    "default": "bfp_bf8",
}

# Mirrors _moe_throughput_galaxy_shard_spec_fn (test_llms.py:1662).
MESH_SHAPE = (4, 8)
MESH_AXIS_NAMES = ("batch", "model")
INPUT_SHARDING = ("batch", None)
KV_CACHE_SHARDING = ("batch", "model", None, None)


# ----------------------------------------------------------------------------
# Layer remap
# ----------------------------------------------------------------------------


def _load_layer18_state_dict(model_name: str, target_layer: int) -> dict:
    """Stream only the safetensors containing layer `target_layer` and return
    a state_dict keyed at ``model.layers.0.*`` (i.e. remapped onto layer 0).

    MoE mxfp4 tensors (gate_up_proj_blocks/_scales, down_proj_blocks/_scales)
    are dequantized into bf16 tensors named gate_up_proj / down_proj.
    """
    index_path = hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    layer_prefix = f"model.layers.{target_layer}."
    relevant = {k: v for k, v in weight_map.items() if k.startswith(layer_prefix)}
    if not relevant:
        raise RuntimeError(f"No keys matching '{layer_prefix}' in the safetensors index")

    # Group keys by shard so we only open each shard once.
    shard_to_keys: dict[str, list[str]] = {}
    for key, shard in relevant.items():
        shard_to_keys.setdefault(shard, []).append(key)
    print(f"[layer18] layer {target_layer} spans {len(shard_to_keys)} shard(s): "
          f"{list(shard_to_keys.keys())}")

    state: dict[str, torch.Tensor] = {}
    mxfp4_blocks: dict[str, torch.Tensor] = {}
    mxfp4_scales: dict[str, torch.Tensor] = {}

    for shard_file, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id=model_name, filename=shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                new_key = "model.layers.0." + key[len(layer_prefix):]

                # Collect mxfp4 halves for later fusion; pass everything else
                # through as-is (bf16 float or bias tensors).
                if new_key.endswith("_blocks"):
                    mxfp4_blocks[new_key[:-len("_blocks")]] = tensor
                elif new_key.endswith("_scales"):
                    mxfp4_scales[new_key[:-len("_scales")]] = tensor
                else:
                    state[new_key] = tensor

    # Fuse mxfp4 pairs into dequantized bf16 tensors.
    for base, blocks in mxfp4_blocks.items():
        scales = mxfp4_scales.pop(base)
        state[base] = convert_moe_packed_tensors(blocks, scales, dtype=torch.bfloat16)
    if mxfp4_scales:
        raise RuntimeError(f"Unmatched mxfp4 scales (no matching blocks): {list(mxfp4_scales)}")

    return state


def load_model_layer18_only(model_name: str, target_layer: int) -> tuple[torch.nn.Module, object]:
    """Instantiate a 1-layer GPT-OSS with layer `target_layer` weights in slot 0."""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.num_hidden_layers = 1
    # layer_types is a plain list on the config; safe to mutate pre-load.
    if hasattr(config, "layer_types") and isinstance(config.layer_types, list):
        config.layer_types = ["full_attention"] * len(config.layer_types)
    # NOTE: _experts_implementation="dense" is not an accepted pre-validation
    # value in this transformers version (valid: eager/grouped_mm/batched_mm).
    # llm_benchmark.setup_model_and_tokenizer sets it AFTER from_pretrained; we do the same.

    print("[layer18] Building 1-layer model (loads only layer 0 from HF)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        quantization_config=Mxfp4Config(dequantize=True),
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
    )
    # Post-load: switch experts to dense static forward (matches llm_benchmark).
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"
    model.eval()

    print(f"[layer18] Streaming layer {target_layer} weights and remapping to slot 0...")
    remap = _load_layer18_state_dict(model_name, target_layer)

    # Assign remapped tensors on top of layer-0 parameters. bfloat16 cast just in case.
    missing, unexpected = model.load_state_dict(remap, strict=False, assign=False)
    # Only "missing" embedding/lm_head/norm keys are expected (not in remap).
    suspect_missing = [k for k in missing if k.startswith("model.layers.0.")]
    if suspect_missing:
        raise RuntimeError(f"Remap missed layer 0 params: {suspect_missing}")
    if unexpected:
        print(f"[layer18] WARNING: unexpected keys during remap: {unexpected[:5]}...")
    print(f"[layer18] Remap complete ({len(remap)} tensors).")

    return model, config


# ----------------------------------------------------------------------------
# StaticCache wrapper so codegen_py sees cache tensors as graph inputs/state
# ----------------------------------------------------------------------------


class ModelWithCache(torch.nn.Module):
    """Internalize StaticCache so codegen_py's tensor-only kwarg filter works.

    StaticCache is a plain Python object; its tensors are NOT moved by
    model.to(device). We replace the cache's per-layer keys/values with
    torch.nn.Buffers on *this* wrapper so that model.to(device) carries them
    to TT device (and xs.mark_sharding can pin them to the KV-cache shard spec).
    """

    def __init__(self, model, config, batch_size: int, max_cache_len: int):
        super().__init__()
        self.inner = model
        self.config = config
        self.cache = StaticCache(
            config=config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.bfloat16,
        )
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.cache.early_initialization(
            batch_size=batch_size,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=torch.bfloat16,
            device="cpu",
        )

        # Expose the cache tensors as buffers so .to(device) moves them.
        self._cache_tensor_names: list[tuple[int, str]] = []
        for i, layer in enumerate(self.cache.layers):
            k_name, v_name = f"cache_keys_{i}", f"cache_values_{i}"
            self.register_buffer(k_name, layer.keys, persistent=False)
            self.register_buffer(v_name, layer.values, persistent=False)
            self._cache_tensor_names.append((i, k_name))
            self._cache_tensor_names.append((i, v_name))

    def _rebind_cache_tensors(self):
        """Point cache.layers[i].keys/values at the (possibly moved) buffers."""
        for i, layer in enumerate(self.cache.layers):
            layer.keys = getattr(self, f"cache_keys_{i}")
            layer.values = getattr(self, f"cache_values_{i}")

    def forward(self, input_ids, cache_position):
        self._rebind_cache_tensors()
        return self.inner(
            input_ids=input_ids,
            past_key_values=self.cache,
            cache_position=cache_position,
            use_cache=True,
        )


# ----------------------------------------------------------------------------
# Sharding (mirrors _moe_throughput_galaxy_shard_spec_fn)
# ----------------------------------------------------------------------------


def apply_moe_galaxy_sharding(model, mesh):
    """Copy of _moe_throughput_galaxy_shard_spec_fn, applied directly."""
    m = model
    xs.mark_sharding(m.model.embed_tokens.weight, mesh, (None, None))
    xs.mark_sharding(m.model.norm.weight, mesh, (None,))
    xs.mark_sharding(m.lm_head.weight, mesh, (None, None))
    for layer in m.model.layers:
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.self_attn.sinks, mesh, ("model",))
        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, None))
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", "batch", None))
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, "batch"))
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", "batch"))
        xs.mark_sharding(layer.input_layernorm.weight, mesh, (None,))
        xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, (None,))


# ----------------------------------------------------------------------------
# Main driver
# ----------------------------------------------------------------------------


def main():
    xr.set_device_type("TT")

    # --- Step 1+2: build 1-layer model with layer 18's weights -------------
    inner_model, config = load_model_layer18_only(MODEL_NAME, TARGET_LAYER)

    # --- Step 3: weight_dtype_overrides BEFORE device transfer -------------
    applied = apply_weight_dtype_overrides(inner_model, WEIGHT_DTYPE_OVERRIDES)
    print(f"[main] Applied {len(applied)} weight_dtype_overrides")
    bfp4_applied = [p for p, d in applied if d == "bfp_bf4"]
    print(f"[main] bfp4-quantized params: {bfp4_applied}")

    # --- Step 4: wrap with cache -------------------------------------------
    wrapped = ModelWithCache(
        inner_model, config,
        batch_size=BATCH_SIZE, max_cache_len=INPUT_SEQUENCE_LENGTH,
    )
    wrapped.eval()

    # --- Step 5: SPMD + device transfer ------------------------------------
    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_devices != MESH_SHAPE[0] * MESH_SHAPE[1]:
        raise RuntimeError(
            f"Expected {MESH_SHAPE[0] * MESH_SHAPE[1]} devices (Galaxy 4x8), "
            f"got {num_devices}. Aborting."
        )
    device_ids = np.arange(num_devices)
    mesh = Mesh(device_ids, MESH_SHAPE, MESH_AXIS_NAMES)

    device = torch_xla.device()
    wrapped = wrapped.to(device, dtype=torch.bfloat16)

    # Mark weight sharding.
    apply_moe_galaxy_sharding(wrapped.inner, mesh)

    # Mark KV-cache sharding (operates on the buffers we registered).
    for i in range(config.num_hidden_layers):
        xs.mark_sharding(getattr(wrapped, f"cache_keys_{i}"), mesh, KV_CACHE_SHARDING)
        xs.mark_sharding(getattr(wrapped, f"cache_values_{i}"), mesh, KV_CACHE_SHARDING)

    # --- Step 6: inputs on device ------------------------------------------
    # Match llm_benchmark.DEFAULT_INPUT_PROMPT so codegen PCC is apples-to-
    # apples with the pytest compile path (which uses the same prompt).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    prompt = "Here is an exaustive list of the best practices for writing clean code:"
    tokens = tok(
        [prompt] * BATCH_SIZE,
        return_tensors="pt", max_length=INPUT_SEQUENCE_LENGTH,
        padding="max_length", truncation=True,
    )["input_ids"]
    assert tokens.shape == (BATCH_SIZE, INPUT_SEQUENCE_LENGTH)
    input_ids = tokens.to(device)
    cache_position = torch.arange(0, INPUT_SEQUENCE_LENGTH).to(device)
    xs.mark_sharding(input_ids, mesh, INPUT_SHARDING)

    # --- Step 7: codegen_py ------------------------------------------------
    os.makedirs(EXPORT_PATH, exist_ok=True)
    print(f"[main] Running codegen_py, export_path={EXPORT_PATH}")
    compile_options = {
        "optimization_level": 1,
        "backend": "codegen_py",
        "export_path": EXPORT_PATH,
        "export_tensors": True,
    }
    torch_xla.set_custom_compile_options(compile_options)
    wrapped.compile(backend="tt", options={"tt_legacy_compile": True})
    with torch.no_grad():
        wrapped(input_ids, cache_position)
    xm.wait_device_ops()
    print("[main] codegen_py completed.")

    # --- Step 8: CPU bypass post-processing --------------------------------
    print(f"[main] Parsing generated code and emitting CPU bypass...")
    result = run_alchemist_cpu_bypass(EXPORT_PATH)
    print(f"[main] bfp4 matmul groups found: {result['groups_found']}")
    if result.get("bypass_path"):
        print(f"[main] CPU-bypassed main: {result['bypass_path']}")
    for g in result.get("groups", []):
        print(f"         {g}")

    print(
        "\n=== NEXT STEP =====================================================\n"
        f"Run both modules and compare their outputs (PCC):\n"
        f"  python tests/benchmark/run_bfp4_layer18_pcc.py {EXPORT_PATH}\n"
        "===================================================================\n"
    )


if __name__ == "__main__":
    main()
