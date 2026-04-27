# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layer-18 codegen driver that matches the pytest prefill inputs exactly.

Same as debug_bfp4_layer18.py but:
  - input_ids = tokenizer(DEFAULT_INPUT_PROMPT) with NO padding  → (batch, 17)
  - StaticCache is still sized to 128 (matches the pytest max_cache_len).
  - EXPORT_PATH = bfp4_layer18_output_pytest_match
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
from transformers import AutoTokenizer

from benchmarks.codegen_accuracy import run_alchemist_cpu_bypass
from tt_torch.weight_dtype import apply_weight_dtype_overrides

from debug_bfp4_layer18 import (
    MODEL_NAME, TARGET_LAYER, BATCH_SIZE, WEIGHT_DTYPE_OVERRIDES,
    MESH_SHAPE, MESH_AXIS_NAMES, INPUT_SHARDING, KV_CACHE_SHARDING,
    ModelWithCache, load_model_layer18_only, apply_moe_galaxy_sharding,
)

MAX_CACHE_LEN = 128  # matches pytest's input_sequence_length default
EXPORT_PATH = str(BENCHMARK_DIR / "bfp4_layer18_output_pytest_match")


def main():
    xr.set_device_type("TT")

    inner_model, config = load_model_layer18_only(MODEL_NAME, TARGET_LAYER)
    applied = apply_weight_dtype_overrides(inner_model, WEIGHT_DTYPE_OVERRIDES)
    print(f"[main] Applied {len(applied)} weight_dtype_overrides")

    # Natural 17-token prefill — same as pytest's construct_inputs (no padding).
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    prompt = "Here is an exaustive list of the best practices for writing clean code:"
    tokens = tok([prompt], return_tensors="pt", max_length=MAX_CACHE_LEN, truncation=True)["input_ids"]
    prefill_len = tokens.shape[1]
    print(f"[main] prefill_len = {prefill_len} (natural tokenisation, no padding)")
    # Broadcast to batch 64.
    input_ids_cpu = tokens.expand(BATCH_SIZE, -1).contiguous()
    cache_position_cpu = torch.arange(0, prefill_len)

    wrapped = ModelWithCache(
        inner_model, config,
        batch_size=BATCH_SIZE, max_cache_len=MAX_CACHE_LEN,
    ).eval()

    xr.use_spmd()
    num_devices = xr.global_runtime_device_count()
    if num_devices != MESH_SHAPE[0] * MESH_SHAPE[1]:
        raise RuntimeError(f"Expected {MESH_SHAPE[0] * MESH_SHAPE[1]} devices, got {num_devices}")
    mesh = Mesh(np.arange(num_devices), MESH_SHAPE, MESH_AXIS_NAMES)
    device = torch_xla.device()
    wrapped = wrapped.to(device, dtype=torch.bfloat16)
    apply_moe_galaxy_sharding(wrapped.inner, mesh)
    for i in range(config.num_hidden_layers):
        xs.mark_sharding(getattr(wrapped, f"cache_keys_{i}"), mesh, KV_CACHE_SHARDING)
        xs.mark_sharding(getattr(wrapped, f"cache_values_{i}"), mesh, KV_CACHE_SHARDING)

    input_ids = input_ids_cpu.to(device)
    cache_position = cache_position_cpu.to(device)
    xs.mark_sharding(input_ids, mesh, INPUT_SHARDING)

    os.makedirs(EXPORT_PATH, exist_ok=True)
    print(f"[main] Running codegen_py, export_path={EXPORT_PATH}")
    torch_xla.set_custom_compile_options({
        "optimization_level": 1,
        "backend": "codegen_py",
        "export_path": EXPORT_PATH,
        "export_tensors": True,
    })
    wrapped.compile(backend="tt", options={"tt_legacy_compile": True})
    with torch.no_grad():
        wrapped(input_ids, cache_position)
    xm.wait_device_ops()
    print("[main] codegen_py completed.")

    # Patch utils.py to FABRIC_1D_RING.
    utils_py = Path(EXPORT_PATH) / "utils.py"
    if utils_py.exists():
        src = utils_py.read_text()
        patched = src.replace(
            "ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)",
            "ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)",
        )
        if patched != src:
            utils_py.write_text(patched)
            print("[main] Patched utils.py -> FABRIC_1D_RING")

    result = run_alchemist_cpu_bypass(EXPORT_PATH)
    print(f"[main] bfp4 matmul groups found: {result['groups_found']}")


if __name__ == "__main__":
    main()
