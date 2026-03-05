"""
Test dispatch/combine backward through gpt_oss A2aSparseMLP.

Replicates the exact torch_model_tester._test_training flow:
  1. Load gpt_oss model (1 layer, a2a_sparse MLP)
  2. torch.compile(inductor) → forward+backward on CPU
  3. zero_grad
  4. torch.compile(backend="tt") → forward+backward on TT
  5. Compare forward and backward PCC

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_dispatch_combine_backward.py 2>&1 | tee out_test.txt
"""

import os
import sys

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant
from third_party.tt_forge_models.training_utils import unpack_forward_output
from tt_torch.sharding import sharding_constraint_tensor


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    if a.numel() == 0:
        return 1.0
    a_mean = a - a.mean()
    b_mean = b - b.mean()
    num = (a_mean * b_mean).sum()
    den = (a_mean.norm() * b_mean.norm()).clamp(min=1e-12)
    return (num / den).item()


def main():
    # ---- SPMD setup ----
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    if n >= 32:
        mesh_shape = (4, 8)
    elif n >= 8:
        mesh_shape = (2, 4)
    else:
        raise RuntimeError(f"Need >=8 devices, got {n}")
    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))

    # ---- Load gpt_oss (1 layer) ----
    loader = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model = loader.load_model()
    inputs = loader.load_inputs()
    model.train()

    print(f"[gpt_oss] Model loaded: {loader.config.num_hidden_layers} layer(s)")
    print(f"[gpt_oss] Input: {inputs['input_ids'].shape}")

    # ==== CPU: forward + backward ====
    cpu_compiled = torch.compile(model, backend="inductor")

    with torch.set_grad_enabled(True):
        cpu_out = cpu_compiled(**inputs)

    cpu_res = unpack_forward_output(cpu_out)
    random_grad = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)

    with torch.set_grad_enabled(True):
        cpu_res.backward(gradient=random_grad)

    cpu_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()

    print(f"[gpt_oss] CPU fwd shape: {cpu_res.shape}")
    print(f"[gpt_oss] CPU grads: {len(cpu_grads)} params")

    # ==== TT: forward + backward (same model, same random_grad) ====
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )

    device = torch_xla.device()
    model.to(device)

    # Shard exactly as loader.py
    shard_specs = loader.load_shard_spec(model)
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    inputs_dev = {k: v.to(device) for k, v in inputs.items()}

    with torch.set_grad_enabled(True):
        tt_out = tt_compiled(**inputs_dev)

    tt_res = unpack_forward_output(tt_out)
    torch_xla.sync(wait=True)

    with torch.set_grad_enabled(True):
        tt_res.backward(gradient=random_grad.to(device))

    # Mark gradient sharding — critical for correct gradient gathering.
    # Matches torch_model_tester._test_training: mark_gradient_sharding().
    for param in model.parameters():
        if param.grad is None or param not in shard_specs:
            continue
        param.grad = sharding_constraint_tensor(
            param.grad, mesh, shard_specs[param],
        )

    wanted = [p.grad for p in model.parameters() if p.grad is not None]
    torch_xla._XLAC._xla_sync_multi(
        wanted, list({p.device.type for p in wanted}), wait=True,
    )

    tt_grads = {n: p.grad.to("cpu") for n, p in model.named_parameters() if p.grad is not None}

    # ==== Compare ====
    fwd_pcc = pcc(cpu_res.detach(), tt_res.detach().to("cpu"))
    print(f"\n[gpt_oss] Forward PCC = {fwd_pcc:.6f}")

    # Per-parameter backward PCC
    mlp_pccs = []
    other_pccs = []
    for name in sorted(cpu_grads.keys()):
        if name not in tt_grads:
            print(f"[gpt_oss] {name}: MISSING on TT")
            continue
        p = pcc(cpu_grads[name], tt_grads[name])
        tag = "mlp" if "mlp" in name else "other"
        if tag == "mlp":
            mlp_pccs.append(p)
        else:
            other_pccs.append(p)
        cn = cpu_grads[name].norm().item()
        tn = tt_grads[name].norm().item()
        print(f"[gpt_oss] {name}: PCC={p:.6f}  cpu_norm={cn:.4f}  tt_norm={tn:.4f}")

    if mlp_pccs:
        print(f"\n[gpt_oss] MLP avg backward PCC = {sum(mlp_pccs)/len(mlp_pccs):.6f}")
    if other_pccs:
        print(f"[gpt_oss] Other avg backward PCC = {sum(other_pccs)/len(other_pccs):.6f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
