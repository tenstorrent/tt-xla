"""Sanity: test the exact Conv3d that hangs in the test runner.

From test runner log (denseunet_3d_detailed.log), the Conv3d inside upsample4
hangs indefinitely. This is:
    Conv3d(224, 96, kernel_size=(3,3,3), padding=(1,1,1))
    Input shape: (1, 224, 16, 128, 128)

This script tests that exact Conv3d in 3 modes:
  Mode 1: Direct eager execution (no torch.compile)
  Mode 2: torch.compile(backend='inductor') - direct call
  Mode 3: Through test runner infrastructure (DeviceRunner + _mask_jax_accelerator)

Expected: all 3 complete in seconds, proving the hang is caused by the
full pytest test runner environment, not the op itself.
"""
import time
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

xr.runtime.set_device_type("TT")

from infra import DeviceConnectorFactory, Framework

DeviceConnectorFactory.create_connector(Framework.JAX)
_torch_connector = DeviceConnectorFactory.create_connector(Framework.TORCH)

# Exact Conv3d from upsample4 in DenseUNet3d
conv = nn.Conv3d(224, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)).eval()
x = torch.randn(1, 224, 16, 128, 128)

# ===== Mode 1: Direct eager =====
print("=" * 60)
print("Mode 1: Conv3d EAGER (no torch.compile)")
print("=" * 60, flush=True)
with torch.no_grad():
    _ = conv(x)  # warmup
    t0 = time.time()
    out = conv(x)
    print(f"DONE in {time.time() - t0:.3f}s  |  shape: {out.shape}", flush=True)

# ===== Mode 2: torch.compile(backend='inductor') =====
print()
print("=" * 60)
print("Mode 2: Conv3d torch.compile(backend='inductor')")
print("=" * 60, flush=True)
compiled_conv = torch.compile(conv, backend="inductor")
with torch.no_grad():
    t0 = time.time()
    out = compiled_conv(x)
    print(f"DONE in {time.time() - t0:.3f}s  |  shape: {out.shape}", flush=True)

# ===== Mode 3: Through test runner infrastructure =====
print()
print("=" * 60)
print("Mode 3: Conv3d via test runner infra")
print("       (compile_torch_workload_for_cpu + DeviceRunner + _mask_jax_accelerator)")
print("=" * 60, flush=True)

torch._dynamo.reset()

conv_c = nn.Conv3d(224, 96, kernel_size=(3, 3, 3), padding=(1, 1, 1)).eval()
x_c = torch.randn(1, 224, 16, 128, 128)

from infra.utilities import compile_torch_workload_for_cpu
from infra.workloads import TorchWorkload
from infra.runners.torch_device_runner import TorchDeviceRunner
from tests.infra.testers.single_chip.model.torch_model_tester import _mask_jax_accelerator

workload = TorchWorkload(model=conv_c, args=[x_c], kwargs={})

print("  compile_torch_workload_for_cpu ...", flush=True)
t0 = time.time()
compile_torch_workload_for_cpu(workload)
print(f"  compile done in {time.time() - t0:.3f}s (lazy)", flush=True)

runner = TorchDeviceRunner(_torch_connector)

print("  run_on_cpu (with _mask_jax_accelerator) ...", flush=True)
t0 = time.time()
with _mask_jax_accelerator():
    out = runner.run_on_cpu(workload)
print(f"  DONE in {time.time() - t0:.3f}s  |  shape: {out.shape}", flush=True)

print()
print("=" * 60)
print("ALL 3 MODES PASSED - Conv3d is NOT the issue.")
print("The hang only occurs in the full pytest test runner environment.")
print("=" * 60, flush=True)
