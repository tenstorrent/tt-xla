"""
Isolate forward PCC: test with MLP zeroed vs attention zeroed.

Usage:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_isolate_fwd.py 2>&1 | tee out_isolate.txt
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant
from third_party.tt_forge_models.training_utils import unpack_forward_output


def pcc(a, b):
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() <= 1:
        return float("nan")
    a -= a.mean(); b -= b.mean()
    denom = a.norm() * b.norm()
    return (a @ b / denom).item() if denom > 0 else float("nan")


def run_test(model, inputs, shard_specs, mesh, label):
    """Run full model: CPU (inductor) then TT, report forward PCC."""
    device = torch_xla.device()

    # CPU forward
    model.eval()
    cpu_compiled = torch.compile(model, backend="inductor")
    with torch.no_grad():
        cpu_out = cpu_compiled(**inputs)
    cpu_res = unpack_forward_output(cpu_out)

    # TT forward
    tt_compiled = torch.compile(model, backend="tt",
                                options={"tt_experimental_compile": False,
                                         "tt_enable_torch_fx_fusion_pass": False})
    # Build name→spec mapping before moving to device (tensor identity changes after .to())
    name_to_spec = {}
    for name, param in model.named_parameters():
        if param in shard_specs:
            name_to_spec[name] = shard_specs[param]
    model.to(device)
    for name, param in model.named_parameters():
        if name in name_to_spec:
            xs.mark_sharding(param, mesh, name_to_spec[name])
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        tt_out = tt_compiled(**inputs_dev)
    tt_res = unpack_forward_output(tt_out)
    torch_xla.sync(wait=True)
    tt_cpu = tt_res.detach().to("cpu")

    p = pcc(cpu_res.detach(), tt_cpu)
    print(f"[{label}] Forward PCC = {p:.6f}  "
          f"cpu_norm={cpu_res.float().norm().item():.2f}  "
          f"tt_norm={tt_cpu.float().norm().item():.2f}")
    return p


class ZeroMLP(nn.Module):
    """Replaces A2aSparseMLP forward with zeros."""
    def __init__(self, original_mlp):
        super().__init__()
        # Keep router so shard_spec lookup still works
        self.router = original_mlp.router
        self.experts = original_mlp.experts

    def forward(self, hidden_states):
        return torch.zeros_like(hidden_states), None


def main():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    mesh_shape = (4, 8) if n >= 32 else (2, 4)
    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))

    # ── Baseline: full model (A2aSparseMLP active) ──
    loader = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model_full = loader.load_model()
    inputs = loader.load_inputs()
    shard_specs = loader.load_shard_spec(model_full)

    print("=== BASELINE (full A2aSparseMLP) ===")
    run_test(model_full, inputs, shard_specs, mesh, "FULL")

    # ── MLP zeroed: isolate attention ──
    loader2 = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model_nomlp = loader2.load_model()
    inputs2 = loader2.load_inputs()
    shard_specs2 = loader2.load_shard_spec(model_nomlp)

    for layer in model_nomlp.model.layers:
        layer.mlp = ZeroMLP(layer.mlp)

    print("\n=== MLP ZEROED (attention only) ===")
    run_test(model_nomlp, inputs2, shard_specs2, mesh, "NO_MLP")

    print("\nDone.")


if __name__ == "__main__":
    main()
