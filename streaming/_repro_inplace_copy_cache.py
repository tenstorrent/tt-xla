"""Test: same template + in-place copy_ on existing device params → cache hit?

Unlike `_repro_weight_swap_cache.py` which replaces `_parameters[name]`
with new XLA tensors (different storage IDs → dynamo cache miss), this
test:
  1. Builds template ONCE with weights on device.
  2. For each subsequent layer, loads CPU weights and uses
     `template_param.data.copy_(new_param.to(device))` to overwrite
     the existing device buffer in-place.
  3. Param Python identity AND device storage location stay the same →
     dynamo cache should hit on subsequent runs.

If t_run2 << t_run1, this is the path forward for layer streaming.
"""
from __future__ import annotations

import gc
import os
import time

import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
    _upload_with_sharding,
)
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from tt_torch.sparse_mlp import enable_sparse_mlp


BATCH = 8
PROMPT_LEN = 128


def malloc_trim():
    try:
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log(tag):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    sys_used = psutil.virtual_memory().used / 1e9
    print(f"[{tag:<38s}] rss={rss:6.2f} sys_used={sys_used:6.2f} GB", flush=True)


def build_model(args):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)


def stream_layer_into(model, layer_id, mesh, mesh_shape, device, args):
    """Load layer_id's HF weights, sparse_mlp rewrite, ship into
    model.layers[layer_id]. Mirrors the existing layer-streaming pipeline.
    """
    block_sd = weight_loader.load_block_state_dict(layer_id)
    prefix = f"layers.{layer_id}."
    stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in block_sd.items()}
    model.layers[layer_id].load_state_dict(stripped, strict=False)
    del block_sd, stripped
    gc.collect()
    enable_sparse_mlp(model.layers[layer_id], mesh=mesh_shape, cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(model.layers[layer_id])
    gc.collect()
    block = model.layers[layer_id]
    specs = _block_shard_spec(block, mesh)
    specs_by_id = {id(t): ps for t, ps in specs.items()}
    del specs
    _ship_module_handle_path(block, specs_by_id, mesh, device,
                             verbose=False, tag=f"l{layer_id}")
    torch_xla.sync(wait=True)
    gc.collect()


def inplace_swap_weights(dst_block, src_block):
    """Copy src_block params/buffers INTO dst_block's existing tensors
    via `.data.copy_()`. Preserves Python identity AND device storage
    location of dst_block tensors → dynamo cache should hit.
    """
    dst_subs = list(dst_block.modules())
    src_subs = list(src_block.modules())
    assert len(dst_subs) == len(src_subs), \
        f"module count mismatch {len(dst_subs)} vs {len(src_subs)}"
    n_params = 0
    n_buffers = 0
    for dst_sub, src_sub in zip(dst_subs, src_subs):
        for name, dst_p in list(dst_sub._parameters.items()):
            src_p = src_sub._parameters.get(name)
            if dst_p is None or src_p is None:
                continue
            assert dst_p.shape == src_p.shape, \
                f"shape mismatch {name}: {dst_p.shape} vs {src_p.shape}"
            dst_p.data.copy_(src_p.data)
            n_params += 1
        for name, dst_b in list(dst_sub._buffers.items()):
            src_b = src_sub._buffers.get(name)
            if dst_b is None or src_b is None:
                continue
            assert dst_b.shape == src_b.shape, \
                f"shape mismatch {name}: {dst_b.shape} vs {src_b.shape}"
            dst_b.data.copy_(src_b.data)
            n_buffers += 1
    return n_params, n_buffers


def main():
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    n = xr.global_runtime_device_count()
    mesh_shape = (2, 4) if n == 8 else (1, n)
    mesh = Mesh(np.arange(n), mesh_shape, ("_axis_0", "_axis_1"))
    device = torch_xla.device()

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = BATCH
    args.max_seq_len = 256
    args.n_layers = 2
    args.compress_ratios = args.compress_ratios[:2]

    log("baseline")

    # Two model instances (CPU initially, will be shipped to device)
    m0 = build_model(args)
    m1 = build_model(args)
    log("post-skeletons")

    print("\n[stream] loading layer 0 into m0 ...", flush=True)
    stream_layer_into(m0, 0, mesh, mesh_shape, device, args)
    log("post-load m0.layers[0]")

    print("[stream] loading layer 1 into m1 ...", flush=True)
    stream_layer_into(m1, 1, mesh, mesh_shape, device, args)
    log("post-load m1.layers[1]")

    # Dummy input
    h = torch.randn(BATCH, PROMPT_LEN, args.hc_mult, args.dim,
                    dtype=torch.bfloat16).to(device)
    xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    input_ids = torch.zeros(BATCH, PROMPT_LEN, dtype=torch.long).to(device)
    xs.mark_sharding(input_ids, mesh, ("_axis_0", None))

    # Use m0.layers[0] as the persistent template.
    template = m0.layers[0]

    @torch.compile(backend="tt")
    def run_block(block, h, sp, input_ids):
        return block(h, sp, input_ids)

    # ---- Run 1: layer 0 weights (cold compile) ----
    print("\n[run 1] template with layer 0 weights (COLD) ...", flush=True)
    log("pre-run-1")
    t0 = time.time()
    h = run_block(template, h, sp, input_ids)
    torch_xla.sync(wait=True)
    t_run1 = time.time() - t0
    print(f"[run 1] {t_run1:.2f}s (cold: dynamo trace + LTC + StableHLO + PJRT compile)",
          flush=True)
    log("post-run-1")

    # ---- Run 2: SAME template, SAME weights — true cache-hit baseline ----
    print("\n[run 2] template UNCHANGED (true cache-hit baseline) ...", flush=True)
    log("pre-run-2")
    t1 = time.time()
    h = run_block(template, h, sp, input_ids)
    torch_xla.sync(wait=True)
    t_run2 = time.time() - t1
    print(f"[run 2] {t_run2:.2f}s (no swap; should be device exec time)",
          flush=True)
    log("post-run-2")

    # ---- In-place swap: template ← m1.layers[1] (preserves identity) ----
    print("\n[swap] template params/buffers ← m1.layers[1] (in-place copy_) ...", flush=True)
    n_p, n_b = inplace_swap_weights(template, m1.layers[1])
    torch_xla.sync(wait=True)
    print(f"[swap] copied {n_p} params, {n_b} buffers", flush=True)
    log("post-swap")

    # ---- Run 3: same template, swapped weights — does cache survive copy_? ----
    print("\n[run 3] template with layer 1 weights (in-place swap) ...", flush=True)
    log("pre-run-3")
    t2 = time.time()
    h = run_block(template, h, sp, input_ids)
    torch_xla.sync(wait=True)
    t_run3 = time.time() - t2
    print(f"[run 3] {t_run3:.2f}s (cache hit if in-place copy_ preserves identity)",
          flush=True)
    log("post-run-3")

    # Sanity
    out = h.detach().to("cpu")
    print(f"\n[final] h[0,0,0,:8] = {out[0,0,0,:8].tolist()}", flush=True)

    print(f"\n[summary]", flush=True)
    print(f"  run1 (cold)        = {t_run1:.2f}s", flush=True)
    print(f"  run2 (no swap)     = {t_run2:.2f}s   (true cache-hit baseline)",
          flush=True)
    print(f"  run3 (in-place swap)= {t_run3:.2f}s", flush=True)
    print(f"  trace-cost(run1-run2) = {t_run1 - t_run2:.2f}s", flush=True)
    print(f"  swap-overhead(run3-run2) = {t_run3 - t_run2:.2f}s", flush=True)
    if t_run3 < t_run2 * 1.2:
        print("[summary] ✓ in-place copy_ preserves cache — viable path!",
              flush=True)
    else:
        print("[summary] ✗ in-place copy_ invalidates cache — same as fresh instance",
              flush=True)


if __name__ == "__main__":
    main()
