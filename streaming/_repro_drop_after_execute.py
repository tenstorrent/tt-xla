"""Test: does host staging drop after first execute (which triggers
ensure_layout)?

Path:
1. Upload a 4 GB CPU tensor via send_cpu_data_to_device (replicated).
   → expect host RAM jump by ~32 GB (8x replicate × 4 GB on TT plugin).
2. Run a trivial op on it (e.g. +0) and torch_xla.sync(wait=True).
   → if ensure_layout fires & releases host owned tensor, RSS drops.
3. del xt.
   → final cleanup; remaining host should drop here at minimum.

If step 2 drops host: TT PJRT releases owned host tensor on ensure_layout.
  → streaming will work; just need to trigger first execute eagerly per
    block, OR accept ~N×13 GB host RAM transient during loading phase.
If step 2 doesn't drop host but step 3 does: TT PJRT plugin keeps owned
  host tensor for the BufferInstance lifetime regardless of execute.
  → need plugin patch to release after first device materialization.

Run: source venv/activate && python streaming/_repro_drop_after_execute.py
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
    mesh_shape = (1, n)
    mesh = xs.Mesh(np.arange(n), mesh_shape, ("a", "b"))
    device = torch_xla.device()

    # 4 GB BF16
    SHAPE = (32768, 65536)

    log("baseline")

    # ---- Step 1: upload WITH sharding from start (streaming pattern) ----
    print("\n--- Step 1: upload with sharding spec (streaming pattern) ---", flush=True)
    t = torch.randn(SHAPE, dtype=torch.bfloat16)
    log("after cpu tensor created")
    spec = xs.ShardingSpec(mesh, ("a", "b"))
    xt = xm.send_cpu_data_to_device([t], device, input_sharding=spec)[0]
    log("after send_cpu_data_to_device(spec)")
    del t
    gc.collect()
    log("after del t")

    # ---- Step 2: trivial op + sync (no extra mark_sharding) ----
    print("\n--- Step 2: trivial op + sync (triggers compile+execute) ---", flush=True)
    log("pre-op")
    out = xt.sum()
    log("after building op (lazy)")
    torch_xla.sync(wait=True)
    log("post-sync (compile+execute)")
    xm.wait_device_ops()
    log("post-wait_device_ops")
    out_cpu = out.detach().to("cpu")
    log("post-out.cpu()")
    print(f"sum result = {out_cpu.item():.2f}", flush=True)

    # ---- Step 3: drop xt ----
    print("\n--- Step 3: del xt ---", flush=True)
    del out, out_cpu, xt
    gc.collect()
    torch_xla.sync(wait=True)
    xm.wait_device_ops()
    log("post-del+sync")


if __name__ == "__main__":
    main()
