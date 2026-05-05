"""Test: torch.func.functional_call with params as explicit input → cache hit?

If params are graph inputs (placeholders) rather than closure-captured
DeviceData nodes, the graph hash should depend only on shape/dtype/device
of the placeholders, not on the actual weight tensor identity.

Pattern:
  @torch.compile(backend="tt")
  def run(template, params, h, sp, ids):
    return torch.func.functional_call(template, params, (h, sp, ids))

  # Layer 0
  h = run(template, layer0_params, h, sp, ids)  # cold

  # Layer 1
  h = run(template, layer1_params, h, sp, ids)  # cache hit?
"""
from __future__ import annotations

import gc
import os
import time

import numpy as np
import psutil
import torch
import torch.func as func
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
    sys_used = psutil.virtual_memory().used / 1e9
    print(f"[{tag:38s}] rss={rss:6.2f} sys_used={sys_used:6.2f} GB", flush=True)


BATCH = 32
PROMPT_LEN = 16


def build_model(args):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        return mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev)


def stream_layer_into(model, layer_id, mesh, mesh_shape, device, args):
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


def collect_block_state(block):
    """Get a dict of all parameters + buffers for functional_call."""
    state = {}
    for name, p in block.named_parameters():
        state[name] = p
    for name, b in block.named_buffers():
        state[name] = b
    return state


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
    args.n_layers = 2
    args.compress_ratios = args.compress_ratios[:2]

    log("baseline")
    m0 = build_model(args)
    m1 = build_model(args)
    log("post-skeletons")

    print("\n[stream] loading layer 0 into m0 ...", flush=True)
    stream_layer_into(m0, 0, mesh, mesh_shape, device, args)
    print("[stream] loading layer 1 into m1 ...", flush=True)
    stream_layer_into(m1, 1, mesh, mesh_shape, device, args)
    log("post-load both")

    h = torch.randn(BATCH, PROMPT_LEN, args.hc_mult, args.dim,
                    dtype=torch.bfloat16).to(device)
    xs.mark_sharding(h, mesh, ("_axis_0", None, None, None))
    sp = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    input_ids = torch.zeros(BATCH, PROMPT_LEN, dtype=torch.long).to(device)
    xs.mark_sharding(input_ids, mesh, ("_axis_0", None))

    template = m0.layers[0]
    params0 = collect_block_state(m0.layers[0])
    params1 = collect_block_state(m1.layers[1])
    print(f"[functional] {len(params0)} params/buffers in dict", flush=True)

    @torch.compile(backend="tt")
    def run_functional(params, h, sp, input_ids):
        # Replace template's params with the provided dict during this call.
        return func.functional_call(template, params, (h, sp, input_ids))

    print("\n[run 1] functional_call with layer 0 params ...", flush=True)
    log("pre-run-1")
    t0 = time.time()
    h = run_functional(params0, h, sp, input_ids)
    torch_xla.sync(wait=True)
    t_run1 = time.time() - t0
    print(f"[run 1] {t_run1:.2f}s", flush=True)
    log("post-run-1")

    print("\n[run 2] functional_call with layer 1 params ...", flush=True)
    log("pre-run-2")
    t1 = time.time()
    h = run_functional(params1, h, sp, input_ids)
    torch_xla.sync(wait=True)
    t_run2 = time.time() - t1
    print(f"[run 2] {t_run2:.2f}s (expect cache hit if functional_call helps)", flush=True)
    log("post-run-2")

    out = h.detach().to("cpu")
    print(f"\n[final] h[0,0,0,:8] = {out[0,0,0,:8].tolist()}", flush=True)

    print(f"\n[summary] run1={t_run1:.2f}s run2={t_run2:.2f}s ratio={t_run2/t_run1:.2f}", flush=True)
    if t_run2 < t_run1 / 5:
        print("[summary] ✓ likely cache hit", flush=True)
    else:
        print("[summary] ✗ cache miss", flush=True)


if __name__ == "__main__":
    main()
