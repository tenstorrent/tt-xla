"""Test: layer-streaming inference — ship layer i, run, release, advance.

Verifies two key invariants:
1. dynamo cache hit: compile cost paid once even if we repeatedly call
   torch.compile-wrapped fn with different `block` instances of same arch.
2. Weight deallocate via Python ref drop: when we replace layer i's
   _parameters with empty, plugin BufferInstance refcount drops, device
   memory frees.

Run: source venv/activate && python streaming/_repro_layer_stream_inference.py
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
from torch import nn
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
)
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from tt_torch.sparse_mlp import enable_sparse_mlp


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


BATCH = 32
PROMPT_LEN = 16


def stream_load_block(model, layer_id, mesh, mesh_shape, device, args):
    """Load block N's weights from HF, sparse-rewrite, ship to device."""
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
                             verbose=False, tag=f"block-{layer_id}")

    apply_weight_dtype_overrides(block, {
        "ffn.mlp.experts.gate_proj": "bfp_bf4",
        "ffn.mlp.experts.up_proj":   "bfp_bf4",
        "ffn.mlp.experts.down_proj": "bfp_bf4",
        "attn.wq_a.weight": "bfp_bf8", "attn.wq_b.weight": "bfp_bf8",
        "attn.wkv.weight":  "bfp_bf8",
        "attn.wo_a.weight": "bfp_bf8", "attn.wo_b.weight": "bfp_bf8",
    })
    torch_xla.sync(wait=True)
    gc.collect()


def release_block(block):
    """Drop all parameter and buffer XLA tensors so plugin BufferInstance
    refcount goes to 0 and device memory frees.

    Replace each Parameter / Buffer with an empty placeholder so the
    Module structure stays intact (we may want to refill it later) but
    the heavy XLA tensors are unreferenced.
    """
    for sub in block.modules():
        for name in list(sub._parameters.keys()):
            sub._parameters[name] = None
        for name in list(sub._buffers.keys()):
            sub._buffers[name] = None
    gc.collect()


def main() -> None:
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
    args.max_seq_len = 32
    args.n_layers = 2
    args.compress_ratios = args.compress_ratios[:2]
    print(f"[args] dim={args.dim} hc_mult={args.hc_mult} layers={args.n_layers}", flush=True)

    log("baseline")

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)
    log("post-skeleton")

    # Build dummy hidden state
    h = torch.randn(BATCH, PROMPT_LEN, args.hc_mult, args.dim,
                    dtype=torch.bfloat16).to(device)
    xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    input_ids = torch.zeros(BATCH, PROMPT_LEN, dtype=torch.long).to(device)
    xs.mark_sharding(input_ids, mesh, ("_axis_0", None))

    # Compile-once helper. dynamo will cache by graph hash; if blocks
    # produce identical graphs (same arch), the binary is reused.
    @torch.compile(backend="tt")
    def run_block(block, h, sp, input_ids):
        return block(h, sp, input_ids)

    # ----- Layer 0: load → run → record output -----
    print("\n[stream] === layer 0 ===", flush=True)
    log("pre-load 0")
    stream_load_block(model, 0, mesh, mesh_shape, device, args)
    log("post-ship 0")

    t0 = time.time()
    h = run_block(model.layers[0], h, sp, input_ids)
    torch_xla.sync(wait=True)
    print(f"[run] layer 0 took {time.time()-t0:.1f}s (compile + execute)", flush=True)
    log("post-execute 0")

    # ----- Release layer 0 weights -----
    release_block(model.layers[0])
    torch_xla.sync(wait=True)
    log("post-release 0")

    # ----- Layer 1: load → run → record -----
    print("\n[stream] === layer 1 ===", flush=True)
    stream_load_block(model, 1, mesh, mesh_shape, device, args)
    log("post-ship 1")

    t1 = time.time()
    h = run_block(model.layers[1], h, sp, input_ids)
    torch_xla.sync(wait=True)
    print(f"[run] layer 1 took {time.time()-t1:.1f}s (cache hit?)", flush=True)
    log("post-execute 1")

    release_block(model.layers[1])
    torch_xla.sync(wait=True)
    log("post-release 1")

    # Pull final h to host for sanity
    out = h.detach().to("cpu")
    print(f"[run] final h[0,0,0,:8] = {out[0,0,0,:8].tolist()}", flush=True)
    log("done")


if __name__ == "__main__":
    main()
