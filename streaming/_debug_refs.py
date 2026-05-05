"""Debug helper: find what's holding CPU tensors after a single block ship.

Usage: source venv/activate && python streaming/_debug_refs.py

Strategy:
1. Run streaming for 1 block.
2. After ship, walk gc.get_objects() filtered to torch.Tensor.
3. Group by device.type — anything CPU > 100 MB is suspicious.
4. For each large CPU tensor, walk gc.get_referrers() to identify what
   holds it.
"""
from __future__ import annotations

import gc
import os
import sys
from collections import defaultdict

import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr

from infra.utilities.torch_multichip_utils import enable_spmd
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
    _top_level_shard_spec, _upload_with_sharding,
)
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from tt_torch.sparse_mlp import enable_sparse_mlp


def rss_gb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def list_large_cpu_tensors(threshold_mb: int = 100):
    """Return [(tensor, size_mb, dtype, shape)] for CPU tensors > threshold."""
    out = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, torch.Tensor) and obj.device.type == "cpu":
                size_mb = obj.element_size() * obj.numel() / 1e6
                if size_mb >= threshold_mb:
                    out.append((obj, size_mb, str(obj.dtype), tuple(obj.shape)))
        except Exception:
            continue
    return out


def describe_referrer(r) -> str:
    t = type(r).__name__
    if isinstance(r, dict):
        return f"dict[len={len(r)}]"
    if isinstance(r, list):
        return f"list[len={len(r)}]"
    if isinstance(r, tuple):
        return f"tuple[len={len(r)}]"
    return t


def find_leak_path(t: torch.Tensor, max_depth: int = 4) -> list:
    """Trace gc.get_referrers chain to find named module/dict that holds t."""
    paths = []
    visited = set()
    queue = [(t, [])]
    while queue:
        obj, path = queue.pop(0)
        if id(obj) in visited or len(path) >= max_depth:
            continue
        visited.add(id(obj))
        try:
            for r in gc.get_referrers(obj):
                if id(r) in visited:
                    continue
                desc = describe_referrer(r)
                new_path = path + [desc]
                # Stop on something descriptive
                if isinstance(r, dict):
                    # Find the key under which obj lives in r
                    keys_for_obj = [k for k, v in r.items() if v is obj]
                    if keys_for_obj:
                        new_path[-1] = f"dict[key={keys_for_obj[0]!r}]"
                if hasattr(r, "__class__") and r.__class__.__module__ != "builtins":
                    new_path.append(f"{r.__class__.__module__}.{r.__class__.__name__}")
                    paths.append(new_path)
                else:
                    queue.append((r, new_path))
        except Exception:
            pass
        if len(paths) > 5:
            break
    return paths[:5]


def main() -> None:
    enable_spmd()
    xr.set_device_type("TT")

    n_devices = xr.global_runtime_device_count()
    if n_devices == 8:
        mesh_shape = (2, 4)
    else:
        mesh_shape = (1, n_devices)
    mesh = xs.Mesh(np.arange(n_devices), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = 32
    args.max_seq_len = 128
    args.n_layers = 1
    args.compress_ratios = args.compress_ratios[:1]

    print(f"\n[debug] n_layers={args.n_layers}", flush=True)

    print(f"[debug] building skeleton ...", flush=True)
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)

    # Skip top-level for simplicity. Focus on block 0.
    print(f"[debug] loading block 0 weights ...", flush=True)
    block_sd = weight_loader.load_block_state_dict(0)
    stripped = {
        (k[len("layers.0."):] if k.startswith("layers.0.") else k): v
        for k, v in block_sd.items()
    }
    model.layers[0].load_state_dict(stripped, strict=False)
    del block_sd, stripped
    gc.collect()

    print(f"[debug] enable_sparse_mlp ...", flush=True)
    enable_sparse_mlp(
        model.layers[0], mesh=mesh_shape, cluster_axis=0,
        config=args, verbose=False,
    )
    _strip_cpu_golden_refs(model.layers[0])
    gc.collect()

    print(f"\n[debug] === BEFORE SHIP === rss={rss_gb():.2f} GB", flush=True)
    by_dtype = defaultdict(lambda: [0, 0.0])  # count, size_mb
    for t, mb, dtype, shape in list_large_cpu_tensors(100):
        by_dtype[dtype][0] += 1
        by_dtype[dtype][1] += mb
    for dtype, (cnt, mb) in sorted(by_dtype.items(), key=lambda x: -x[1][1]):
        print(f"[debug]   {dtype:25s} count={cnt:3d} total={mb/1024:7.2f} GB", flush=True)

    print(f"\n[debug] shipping block 0 ...", flush=True)
    block = model.layers[0]
    block_specs = _block_shard_spec(block, mesh)
    block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
    del block_specs
    _ship_module_handle_path(
        block, block_specs_by_id, mesh, device, verbose=True, tag="block-0",
    )
    torch_xla.sync(wait=True)
    import torch_xla.core.xla_model as xm
    xm.wait_device_ops()
    gc.collect()

    print(f"\n[debug] === AFTER SHIP === rss={rss_gb():.2f} GB", flush=True)
    by_dtype = defaultdict(lambda: [0, 0.0])
    large_cpu = list_large_cpu_tensors(100)
    for t, mb, dtype, shape in large_cpu:
        by_dtype[dtype][0] += 1
        by_dtype[dtype][1] += mb
    for dtype, (cnt, mb) in sorted(by_dtype.items(), key=lambda x: -x[1][1]):
        print(f"[debug]   {dtype:25s} count={cnt:3d} total={mb/1024:7.2f} GB", flush=True)

    # For the top 5 largest CPU tensors AFTER ship, trace referrers.
    print(f"\n[debug] === LEAK PATHS for largest CPU tensors after ship ===", flush=True)
    large_cpu.sort(key=lambda x: -x[1])
    for i, (t, mb, dtype, shape) in enumerate(large_cpu[:5]):
        ptr = t.data_ptr()
        rc = sys.getrefcount(t) - 1  # subtract local
        print(f"\n[debug] [{i}] tensor: dtype={dtype} shape={shape} size={mb:.1f} MB", flush=True)
        print(f"[debug]   data_ptr=0x{ptr:x} refcount={rc} is_leaf={t.is_leaf} grad={t.requires_grad}", flush=True)
        # Direct gc.get_referrers (no path) — show types and brief desc
        try:
            referrers = gc.get_referrers(t)
            print(f"[debug]   {len(referrers)} direct referrers", flush=True)
            for j, r in enumerate(referrers[:10]):
                t_class = type(r).__name__
                if isinstance(r, dict):
                    keys_for_obj = [k for k, v in r.items() if v is t]
                    sample_keys = list(r.keys())[:3]
                    print(f"[debug]     [{j}] dict keys_holding_target={keys_for_obj} sample_keys={sample_keys}", flush=True)
                elif isinstance(r, list):
                    indices = [k for k, v in enumerate(r) if v is t]
                    print(f"[debug]     [{j}] list len={len(r)} indices_holding_target={indices}", flush=True)
                elif isinstance(r, tuple):
                    indices = [k for k, v in enumerate(r) if v is t]
                    elem_types = [type(x).__name__ for x in r]
                    elem_repr = []
                    for k, x in enumerate(r):
                        if isinstance(x, torch.Tensor):
                            elem_repr.append(f"Tensor{tuple(x.shape)}")
                        elif isinstance(x, (int, float, str, bool, type(None))):
                            elem_repr.append(repr(x)[:30])
                        else:
                            elem_repr.append(type(x).__name__)
                    print(f"[debug]     [{j}] tuple len={len(r)} indices_holding_target={indices} elems={elem_repr}", flush=True)
                    # Find what holds this tuple
                    tuple_referrers = gc.get_referrers(r)
                    for k, tr in enumerate(tuple_referrers[:3]):
                        t_class = type(tr).__name__
                        mod_name = tr.__class__.__module__ if hasattr(tr, '__class__') else '?'
                        print(f"[debug]         tuple_holder[{k}] {mod_name}.{t_class}", flush=True)
                elif r.__class__.__module__ == 'builtins' and t_class == 'frame':
                    print(f"[debug]     [{j}] FRAME f_code={r.f_code.co_name} f_lineno={r.f_lineno}", flush=True)
                else:
                    print(f"[debug]     [{j}] {r.__class__.__module__}.{t_class}", flush=True)
        except Exception as e:
            print(f"[debug]   error getting referrers: {e}", flush=True)
        paths = find_leak_path(t)
        if paths:
            for j, path in enumerate(paths):
                print(f"[debug]   path[{j}]: {' -> '.join(path)}", flush=True)


if __name__ == "__main__":
    main()
