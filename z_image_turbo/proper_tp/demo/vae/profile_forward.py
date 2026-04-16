#!/usr/bin/env python3
"""Profile the VAE forward pass — split consteval vs actual forward time."""

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch
import ttnn

DRAM_RM = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM, None)

import utils
import main as generated_main

print("Loading static inputs ...")
orig = os.getcwd()
os.chdir(_HERE)
try:
    inputs = generated_main.load_inputs_for__main()
finally:
    os.chdir(orig)
print(f"Loaded {len(inputs)} inputs")

mesh_device = utils.DeviceGetter.get_device((1, 4))

# Prepare a dummy latent
raw = torch.randn(1, 16, 64, 64)
z = (raw.float() / 0.3611) + 0.1159
z_tt = ttnn.from_torch(
    z.bfloat16(), dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.Layout.ROW_MAJOR, device=mesh_device,
    memory_config=DRAM_RM,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
inputs[124] = z_tt

# ── Time just the consteval ──────────────────────────────────────────────────
print("\nTiming consteval only ...")
generated_main._cached__main = {}
t0 = time.perf_counter()
generated_main.consteval__main(generated_main._cached__main, inputs)
t_consteval = time.perf_counter() - t0
print(f"  consteval: {t_consteval:.2f} s")

# ── Time just the forward pass (cache already warm) ──────────────────────────
# Re-prepare latent (was consumed)
z_tt2 = ttnn.from_torch(
    z.bfloat16(), dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.Layout.ROW_MAJOR, device=mesh_device,
    memory_config=DRAM_RM,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
inputs[124] = z_tt2

print("Timing first forward pass (consteval already done, JIT cold) ...")
t0 = time.perf_counter()
outputs = generated_main._main(inputs)
t_forward1 = time.perf_counter() - t0
print(f"  1st forward: {t_forward1:.2f} s  ← JIT compilation here")

# ── Reload static inputs and time second forward pass ────────────────────────
orig = os.getcwd()
os.chdir(_HERE)
try:
    inputs2 = generated_main.load_inputs_for__main()
finally:
    os.chdir(orig)

z_tt3 = ttnn.from_torch(
    z.bfloat16(), dtype=ttnn.DataType.BFLOAT16,
    layout=ttnn.Layout.ROW_MAJOR, device=mesh_device,
    memory_config=DRAM_RM,
    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
)
inputs2[124] = z_tt3

print("Timing second forward pass (JIT hot) ...")
t0 = time.perf_counter()
outputs2 = generated_main._main(inputs2)
t_forward2 = time.perf_counter() - t0
print(f"  2nd forward: {t_forward2:.2f} s")

print(f"\nSummary:")
print(f"  consteval:    {t_consteval:.1f} s")
print(f"  1st forward:  {t_forward1:.1f} s  ← Metalium kernel JIT")
print(f"  2nd forward:  {t_forward2:.3f} s  ← kernel cache hot")
print(f"  Total warmup: {t_consteval + t_forward1:.1f} s")
