"""Test: can we compile and execute a single Block standalone?

If yes, this opens the door to layer-streaming inference where each
forward step:
  1. ships layer N's weights to device
  2. runs `compiled_block(h, sp, ids)` for that layer
  3. deallocates layer N's weights, moves on to N+1

We're checking only step 1+2 (the mechanics) here.

Run: source venv/activate && python streaming/_repro_per_layer_compile.py
"""
from __future__ import annotations

import gc
import os

import numpy as np
import psutil
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from infra.utilities.torch_multichip_utils import enable_spmd
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
    _top_level_shard_spec, _upload_with_sharding,
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


# Block has shape:
#   x: [b, s, hc_mult, d]
#   start_pos: scalar
#   input_ids: [b, s]
BATCH = 32
PROMPT_LEN = 16


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
    args.n_layers = 1  # only need block 0 for this test
    args.compress_ratios = args.compress_ratios[:1]
    print(f"[args] n_layers={args.n_layers} dim={args.dim} hc_mult={args.hc_mult}", flush=True)

    log("baseline")

    # Build skeleton (we need top-level for embed-equivalent dummy input).
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)
    log("post-skeleton")

    # Just load block 0 weights (skip top-level since we'll feed synthetic input).
    block_sd = weight_loader.load_block_state_dict(0)
    stripped = {(k[len("layers.0."):] if k.startswith("layers.0.") else k): v
                for k, v in block_sd.items()}
    model.layers[0].load_state_dict(stripped, strict=False)
    del block_sd, stripped
    gc.collect()
    log("post-block-0-load")

    enable_sparse_mlp(model.layers[0], mesh=mesh_shape, cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(model.layers[0])
    gc.collect()
    log("post-sparse-rewrite")

    block = model.layers[0]
    block_specs = _block_shard_spec(block, mesh)
    block_specs_by_id = {id(t): ps for t, ps in block_specs.items()}
    del block_specs
    _ship_module_handle_path(block, block_specs_by_id, mesh, device,
                             verbose=True, tag="block-0")
    torch_xla.sync(wait=True)
    gc.collect()
    log("post-block-0-ship")

    # Apply BFP overrides (lazy, parametrize).
    overrides = {
        "ffn.mlp.experts.gate_proj": "bfp_bf4",
        "ffn.mlp.experts.up_proj":   "bfp_bf4",
        "ffn.mlp.experts.down_proj": "bfp_bf4",
        "attn.wq_a.weight": "bfp_bf8",
        "attn.wq_b.weight": "bfp_bf8",
        "attn.wkv.weight":  "bfp_bf8",
        "attn.wo_a.weight": "bfp_bf8",
        "attn.wo_b.weight": "bfp_bf8",
    }
    applied = apply_weight_dtype_overrides(block, overrides)
    print(f"[wdtype] applied {len(applied)}", flush=True)
    log("post-wdtype")

    # Build dummy input matching Block.forward signature: x, start_pos, input_ids
    # x: [B, S, hc_mult, dim]
    x = torch.randn(BATCH, PROMPT_LEN, args.hc_mult, args.dim, dtype=torch.bfloat16).to(device)
    xs.mark_sharding(x, mesh, ("_axis_0", None, None, None))
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    input_ids = torch.zeros(BATCH, PROMPT_LEN, dtype=torch.long).to(device)
    xs.mark_sharding(input_ids, mesh, ("_axis_0", None))

    log("pre-compile")

    # Standalone compile of single Block
    compiled = torch.compile(block, backend="tt")

    print("[run] compiling + executing single block ...", flush=True)
    out = compiled(x, sp, input_ids)
    torch_xla.sync(wait=True)
    log("post-execute")

    print(f"[run] output shape = {out.shape}", flush=True)
    out_cpu = out.detach().to("cpu")
    print(f"[run] output[0,0,0,:8] = {out_cpu[0,0,0,:8].tolist()}", flush=True)
    log("post-result")


if __name__ == "__main__":
    main()
