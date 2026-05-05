"""Test: does .cpu() on a streaming-uploaded XLA tensor trigger
ensure_layout (releasing plugin host staging)?

Three scenarios:
1. baseline: send + del → measure host RAM (should still hold plugin staging)
2. .cpu() trigger: send + del + .cpu() → measure host RAM (does it drop?)
3. synthetic op + sync: send + del + (xt + 0).sum() + sync → measure (should drop)

Run: source venv/activate && python streaming/_repro_cpu_trigger.py
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


def malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log(tag: str) -> None:
    malloc_trim()
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    sys_used = psutil.virtual_memory().used / 1e9
    print(f"[{tag:38s}] rss={rss:6.2f} sys_used={sys_used:6.2f} GB", flush=True)


def main() -> None:
    enable_spmd()
    xr.set_device_type("TT")

    n = xr.global_runtime_device_count()
    mesh = xs.Mesh(np.arange(n), (1, n), ("a", "b"))
    device = torch_xla.device()
    SHAPE = (32768, 65536)  # 4 GB BF16

    log("baseline")

    # Test A: 3 tensors, .cpu() ONLY on first one
    print("\n--- Test A: 3 tensors, .cpu() on first only ---", flush=True)
    spec = xs.ShardingSpec(mesh, ("a", "b"))
    t1 = torch.randn(SHAPE, dtype=torch.bfloat16)
    xt1 = xm.send_cpu_data_to_device([t1], device, input_sharding=spec)[0]
    del t1
    log("A: post send xt1")
    t2 = torch.randn(SHAPE, dtype=torch.bfloat16)
    xt2 = xm.send_cpu_data_to_device([t2], device, input_sharding=spec)[0]
    del t2
    log("A: post send xt2")
    t3 = torch.randn(SHAPE, dtype=torch.bfloat16)
    xt3 = xm.send_cpu_data_to_device([t3], device, input_sharding=spec)[0]
    del t3
    gc.collect()
    log("A: post send xt3 (3 tensors uploaded)")

    # Trigger .cpu() ONLY on xt1
    cpu1 = xt1.cpu()
    del cpu1
    gc.collect()
    log("A: post xt1.cpu() + del — does xt2,xt3 staging also release?")


if __name__ == "__main__":
    main()
