"""Test: how long does Block execute take when cache is GUARANTEED hit?

Pattern: same template, same weights, multiple sequential calls.
First call = cold compile + execute.
Subsequent calls = pure execute (no compile).

If repeat ~~ first → no cache, recompiling every time.
If repeat << first → cache hit, the first call's overhead = compile.
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
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
)
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from tt_torch.sparse_mlp import enable_sparse_mlp


def malloc_trim():
    try:
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def log(tag):
    malloc_trim()
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    print(f"[{tag:38s}] rss={rss:6.2f} GB", flush=True)


BATCH = 32
PROMPT_LEN = 16


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
    args.max_seq_len = 32
    args.n_layers = 1
    args.compress_ratios = args.compress_ratios[:1]

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)

    block_sd = weight_loader.load_block_state_dict(0)
    stripped = {(k[len("layers.0."):] if k.startswith("layers.0.") else k): v
                for k, v in block_sd.items()}
    model.layers[0].load_state_dict(stripped, strict=False)
    del block_sd, stripped
    gc.collect()
    enable_sparse_mlp(model.layers[0], mesh=mesh_shape, cluster_axis=0,
                     config=args, verbose=False)
    _strip_cpu_golden_refs(model.layers[0])
    block = model.layers[0]
    specs = _block_shard_spec(block, mesh)
    specs_by_id = {id(t): ps for t, ps in specs.items()}
    del specs
    _ship_module_handle_path(block, specs_by_id, mesh, device, verbose=False, tag="b0")
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

    h = torch.randn(BATCH, PROMPT_LEN, args.hc_mult, args.dim,
                    dtype=torch.bfloat16).to(device)
    xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    input_ids = torch.zeros(BATCH, PROMPT_LEN, dtype=torch.long).to(device)
    xs.mark_sharding(input_ids, mesh, ("_axis_0", None))

    @torch.compile(backend="tt")
    def run_block(block, h, sp, input_ids):
        return block(h, sp, input_ids)

    # Repeat 4 times with same template + same weights.
    log("pre")
    for i in range(4):
        t0 = time.time()
        h = run_block(block, h, sp, input_ids)
        torch_xla.sync(wait=True)
        elapsed = time.time() - t0
        print(f"[run {i}] {elapsed:.2f}s", flush=True)
        # don't accumulate h beyond a single layer
        if i == 0:
            ref = h
        log(f"post-run-{i}")


if __name__ == "__main__":
    main()
