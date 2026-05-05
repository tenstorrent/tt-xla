"""Probe: what releases device DRAM after a torch.compile call?

Setup: small fn calling block(h). Run twice with different weights.
Measure device DRAM via TT_RUNTIME_MEMORY_LOG_LEVEL=program.

Try several release mechanisms and observe:
  - just gc.collect + sync
  - torch._dynamo.reset()
  - _xla_clear_pending_irs

Run: source venv/activate && \\
     TT_RUNTIME_MEMORY_LOG_LEVEL=program \\
     python streaming/_repro_dram_release.py
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
from tt_torch.sparse_mlp import enable_sparse_mlp
from tests.torch.models.deepseek_v4 import weight_loader
from streaming.streaming_loader import (
    _block_shard_spec, _ship_module_handle_path, _strip_cpu_golden_refs,
)
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)
from torch import nn


def _malloc_trim() -> None:
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def host_log(tag):
    _malloc_trim()
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss / 1e9
    print(f"\n###### {tag} | host rss={rss:.2f} GB ######", flush=True)


def banner(tag):
    print(f"\n>>>>>> {tag} <<<<<<", flush=True)


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
    args.max_batch_size = 32
    args.n_layers = 1
    args.compress_ratios = (0,)
    args.max_seq_len = 32

    BATCH = 32
    PROMPT_LEN = 16

    def build_block_with_weights(layer_id):
        prev = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            m = mdo.Transformer(args).eval()
        finally:
            torch.set_default_dtype(prev)
        sd = weight_loader.load_block_state_dict(layer_id)
        prefix = f"layers.{layer_id}."
        stripped = {(k[len(prefix):] if k.startswith(prefix) else k): v
                    for k, v in sd.items()}
        m.layers[0].load_state_dict(stripped, strict=False)
        del sd, stripped
        gc.collect()
        enable_sparse_mlp(m.layers[0], mesh=mesh_shape, cluster_axis=0,
                         config=args, verbose=False)
        _strip_cpu_golden_refs(m.layers[0])
        block = m.layers[0]
        specs = _block_shard_spec(block, mesh)
        specs_by_id = {id(t): ps for t, ps in specs.items()}
        del specs
        _ship_module_handle_path(block, specs_by_id, mesh, device,
                                 verbose=False, tag=f"l{layer_id}")
        torch_xla.sync(wait=True)
        gc.collect()
        return m, block

    # Inputs
    h = torch.randn(BATCH, PROMPT_LEN, args.hc_mult, args.dim,
                    dtype=torch.bfloat16).to(device)
    xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    input_ids = torch.zeros(BATCH, PROMPT_LEN, dtype=torch.long).to(device)
    xs.mark_sharding(input_ids, mesh, ("_axis_0", None))

    @torch.compile(backend="tt")
    def run_block(block, h, sp, ids):
        return block(h, sp, ids)

    # Run with layer 0 weights
    banner("Build block with layer 0 weights")
    m0, block0 = build_block_with_weights(0)
    host_log("after build m0")
    banner("EXEC layer 0 first time (cold)")
    out = run_block(block0, h, sp, input_ids)
    torch_xla.sync(wait=True)
    host_log("after exec m0")

    banner("EXEC layer 0 second time (cache hit, same instance)")
    out = run_block(block0, h, sp, input_ids)
    torch_xla.sync(wait=True)
    host_log("after exec m0 again")

    # Try clear pending IRs
    banner("CLEAR PENDING IRs")
    torch_xla._XLAC._clear_pending_irs(str(device))
    gc.collect()
    torch_xla.sync(wait=True)
    host_log("after clear pending irs")

    # Try _dynamo.reset
    banner("DYNAMO RESET")
    torch._dynamo.reset()
    gc.collect()
    torch_xla.sync(wait=True)
    host_log("after dynamo reset")

    # Drop block0 + m0 entirely
    banner("DEL m0 / block0 / out")
    del out, block0, m0
    gc.collect()
    torch_xla.sync(wait=True)
    host_log("after del")

    # Build with layer 1 weights and run
    banner("Build block with layer 1 weights")
    m1, block1 = build_block_with_weights(1)
    host_log("after build m1")
    banner("EXEC layer 1 (after dynamo reset)")
    out = run_block(block1, h, sp, input_ids)
    torch_xla.sync(wait=True)
    host_log("after exec m1")


if __name__ == "__main__":
    main()
