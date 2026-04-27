# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end bfp4 CPU-bypass accuracy check for the full GPT-OSS-120B model.

Same pipeline as debug_bfp4_layer18.py but builds the full 36-layer model
via the standard HF load (no layer remap). Every bfp4 matmul (72 of them —
one gate_up_proj + one down_proj per layer) is replaced with a CPU
torch.matmul. This takes MUCH longer than the single-layer version: CPU
matmul on layer-18-sized MoE weights was ~8 s each; 72 of them per step
plus all-gather overhead → expect 15–30 min for the bypass run.

Outputs
-------
    tests/benchmark/bfp4_full_output/main.py
    tests/benchmark/bfp4_full_output/main_cpu_bypass.py
    tests/benchmark/bfp4_full_output/tensors/ (~280 GB for real HF weights)
    tests/benchmark/bfp4_full_output/device_outputs.pt
    tests/benchmark/bfp4_full_output/bypass_outputs.pt
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = REPO_ROOT / "tests" / "benchmark"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BENCHMARK_DIR))

os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers.cache_utils import StaticCache

from benchmarks.codegen_accuracy import run_alchemist_cpu_bypass
from benchmarks.llm_benchmark import setup_model_and_tokenizer
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant
from tt_torch.weight_dtype import apply_weight_dtype_overrides

# ---- Config (mirror test_gpt_oss_120b_tp_galaxy_batch_size_64) --------------
MODEL_VARIANT = ModelVariant.GPT_OSS_120B
# batch=64 is the pytest config but hits DRAM OOM in the codegen_py path;
# batch=16 is the smallest power-of-two that still exercises all MoE experts
# across a 64-chip-layout topic distribution and leaves headroom for
# intermediate activations.
BATCH_SIZE = 16
INPUT_SEQUENCE_LENGTH = 128
EXPORT_PATH = str(BENCHMARK_DIR / "bfp4_full_output_b16_opt2")

WEIGHT_DTYPE_OVERRIDES = {
    "model.layers.*.mlp.router.weight": "bf16",
    "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
    "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
    "default": "bfp_bf8",
}

MESH_SHAPE = (4, 8)
MESH_AXIS_NAMES = ("batch", "model")
INPUT_SHARDING = ("batch", None)
KV_CACHE_SHARDING = ("batch", "model", None, None)


class ModelWithCache(torch.nn.Module):
    """Internalise StaticCache so codegen_py's tensor-only kwarg filter works."""

    def __init__(self, model, config, batch_size: int, max_cache_len: int):
        super().__init__()
        self.inner = model
        self.config = config
        self.cache = StaticCache(
            config=config, max_batch_size=batch_size, max_cache_len=max_cache_len,
            device="cpu", dtype=torch.bfloat16,
        )
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.cache.early_initialization(
            batch_size=batch_size, num_heads=num_kv_heads, head_dim=head_dim,
            dtype=torch.bfloat16, device="cpu",
        )
        for i, layer in enumerate(self.cache.layers):
            self.register_buffer(f"cache_keys_{i}", layer.keys, persistent=False)
            self.register_buffer(f"cache_values_{i}", layer.values, persistent=False)

    def _rebind_cache_tensors(self):
        for i, layer in enumerate(self.cache.layers):
            layer.keys = getattr(self, f"cache_keys_{i}")
            layer.values = getattr(self, f"cache_values_{i}")

    def forward(self, input_ids, cache_position):
        self._rebind_cache_tensors()
        return self.inner(
            input_ids=input_ids, past_key_values=self.cache,
            cache_position=cache_position, use_cache=True,
        )


def _moe_throughput_galaxy_shard_spec_fn(model, mesh):
    """Direct copy of the sharding spec from test_gpt_oss_120b_tp_galaxy_batch_size_64."""
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


def main():
    xr.set_device_type("TT")

    print("[main] Loading full GPT-OSS-120B (all 36 layers)...")
    model_loader = ModelLoader(variant=MODEL_VARIANT)
    inner_model, tokenizer = setup_model_and_tokenizer(model_loader, MODEL_VARIANT)
    config = inner_model.config
    print(f"[main] num_hidden_layers = {config.num_hidden_layers}")

    applied = apply_weight_dtype_overrides(inner_model, WEIGHT_DTYPE_OVERRIDES)
    print(f"[main] Applied {len(applied)} weight_dtype_overrides")
    bfp4 = [p for p, d in applied if d == "bfp_bf4"]
    print(f"[main] bfp4-quantized params: {len(bfp4)} (expect 2 per layer × 36 = 72)")

    wrapped = ModelWithCache(
        inner_model, config,
        batch_size=BATCH_SIZE, max_cache_len=INPUT_SEQUENCE_LENGTH,
    ).eval()

    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_devices != MESH_SHAPE[0] * MESH_SHAPE[1]:
        raise RuntimeError(
            f"Expected {MESH_SHAPE[0] * MESH_SHAPE[1]} devices, got {num_devices}"
        )
    mesh = Mesh(np.arange(num_devices), MESH_SHAPE, MESH_AXIS_NAMES)

    device = torch_xla.device()
    wrapped = wrapped.to(device, dtype=torch.bfloat16)
    _moe_throughput_galaxy_shard_spec_fn(wrapped.inner, mesh)
    for i in range(config.num_hidden_layers):
        xs.mark_sharding(getattr(wrapped, f"cache_keys_{i}"), mesh, KV_CACHE_SHARDING)
        xs.mark_sharding(getattr(wrapped, f"cache_values_{i}"), mesh, KV_CACHE_SHARDING)

    input_ids = torch.zeros(BATCH_SIZE, INPUT_SEQUENCE_LENGTH, dtype=torch.long).to(device)
    cache_position = torch.arange(0, INPUT_SEQUENCE_LENGTH).to(device)
    xs.mark_sharding(input_ids, mesh, INPUT_SHARDING)

    os.makedirs(EXPORT_PATH, exist_ok=True)
    print(f"[main] Running codegen_py, export_path={EXPORT_PATH}")
    torch_xla.set_custom_compile_options({
        "optimization_level": 2,
        "backend": "codegen_py",
        "export_path": EXPORT_PATH,
        "export_tensors": True,
    })
    wrapped.compile(backend="tt", options={"tt_legacy_compile": True})
    with torch.no_grad():
        wrapped(input_ids, cache_position)
    xm.wait_device_ops()
    print("[main] codegen_py completed.")

    # Patch generated utils.py to use FABRIC_1D_RING (matches PJRT runtime's
    # auto-selection on Galaxy; see mesh_fabric_config.cpp::classifyAxis).
    utils_py = Path(EXPORT_PATH) / "utils.py"
    if utils_py.exists():
        src = utils_py.read_text()
        patched = src.replace(
            "ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)",
            "ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)",
        )
        if patched != src:
            utils_py.write_text(patched)
            print("[main] Patched utils.py: FABRIC_1D -> FABRIC_1D_RING")

    print(f"[main] Parsing generated code and emitting CPU bypass...")
    result = run_alchemist_cpu_bypass(EXPORT_PATH)
    print(f"[main] bfp4 matmul groups found: {result['groups_found']}")
    print(f"[main] (expect 72 = 2 per layer × 36 layers)")

    print(
        "\n=== NEXT STEP =====================================================\n"
        f"  python tests/benchmark/run_bfp4_layer18_pcc.py {EXPORT_PATH}\n"
        f"  python tests/benchmark/recompute_pcc.py {EXPORT_PATH}\n"
        "===================================================================\n"
    )


if __name__ == "__main__":
    main()
