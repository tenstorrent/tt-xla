"""Minimal reproducer: does `send_cpu_data_to_device` actually release
the source CPU tensor's storage?

Compares two paths:
  A) `t.to(xla_device)` — known-broken (keeps tensor_data shadow)
  B) `xm.send_cpu_data_to_device([t], xla_device, input_sharding=spec)`
     — handle-only path, expected to release after wait_device_ops().

Run: source venv/activate && python streaming/_repro_release.py
"""
from __future__ import annotations

import gc
import os

import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

from infra.utilities.torch_multichip_utils import enable_spmd


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def sys_used_gb() -> float:
    return psutil.virtual_memory().used / 1e9


def malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log(tag: str) -> None:
    malloc_trim()
    print(f"[{tag:30s}] rss={rss_gb():6.2f} sys_used={sys_used_gb():6.2f} GB", flush=True)


def main() -> None:
    enable_spmd()
    xr.set_device_type("TT")

    n_devices = xr.global_runtime_device_count()
    mesh_shape = (1, n_devices)
    mesh = xs.Mesh(np.arange(n_devices), mesh_shape, ("a", "b"))
    device = torch_xla.device()

    SHAPE = (32768, 65536)  # 4 GB BF16 — big enough to dominate noise

    log("baseline")

    # ---------------------- Path A: `.to(device)` ----------------------
    print("\n--- A) t.to(device) ---", flush=True)
    t_a = torch.randn(SHAPE, dtype=torch.bfloat16)
    log("A: cpu tensor created")
    xt_a = t_a.to(device)
    log("A: after .to(device)")
    del t_a  # drop our Python ref to the CPU tensor
    gc.collect()
    log("A: del t_a + gc")
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("A: post sync+wait_device_ops")
    del xt_a
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("A: del xt_a + sync")

    # ---------------------- Path B: send_cpu_data_to_device ----------------------
    print("\n--- B) send_cpu_data_to_device ---", flush=True)
    t_b = torch.randn(SHAPE, dtype=torch.bfloat16)
    log("B: cpu tensor created")
    spec = xs.ShardingSpec(mesh, ("a", "b"))
    xt_b = xm.send_cpu_data_to_device([t_b], device, input_sharding=spec)[0]
    log("B: after send_cpu_data_to_device")
    del t_b
    gc.collect()
    log("B: del t_b + gc")
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("B: post sync+wait_device_ops")
    del xt_b
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("B: del xt_b + sync")

    # ---------------------- Path C: replicated (no sharding) ----------------------
    print("\n--- C) send_cpu_data_to_device replicated ---", flush=True)
    t_c = torch.randn(SHAPE, dtype=torch.bfloat16)
    log("C: cpu tensor created")
    xt_c = xm.send_cpu_data_to_device([t_c], device, input_sharding=None)[0]
    log("C: after send_cpu_data_to_device")
    del t_c
    gc.collect()
    log("C: del t_c + gc")
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("C: post sync+wait_device_ops")
    del xt_c
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("C: del xt_c + sync")


if __name__ == "__main__":
    main()
