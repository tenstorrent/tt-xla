"""Test dispatch/combine backward through gpt_oss A2aSparseMLP with diagnostics."""
import os, sys, numpy as np, torch, torch_xla
import torch_xla.distributed.spmd as xs, torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant
from third_party.tt_forge_models.training_utils import unpack_forward_output
from tt_torch.sharding import sharding_constraint_tensor

def pcc(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    if a.numel() == 0: return 1.0
    am, bm = a - a.mean(), b - b.mean()
    return ((am * bm).sum() / (am.norm() * bm.norm()).clamp(min=1e-12)).item()

def main():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd(); xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()
    n = xr.global_runtime_device_count()
    mesh_shape = (4, 8) if n >= 32 else (2, 4) if n >= 8 else None
    assert mesh_shape, "Need >=8 devices"
    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))

    loader = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model = loader.load_model(); inputs = loader.load_inputs(); model.train()
    print("[info] loaded")

    # CPU
    cpu_compiled = torch.compile(model, backend="inductor")
    with torch.set_grad_enabled(True):
        cpu_res = unpack_forward_output(cpu_compiled(**inputs))
    rg = torch.randn(cpu_res.shape, dtype=cpu_res.dtype)
    with torch.set_grad_enabled(True):
        cpu_res.backward(gradient=rg)
    cpu_grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    print("[info] cpu grads:", len(cpu_grads))

    # TT
    tt_compiled = torch.compile(model, backend="tt",
        options={"tt_experimental_compile": False, "tt_enable_torch_fx_fusion_pass": False})
    device = torch_xla.device(); model.to(device)
    shard_specs = loader.load_shard_spec(model)
    for t, s in shard_specs.items(): xs.mark_sharding(t, mesh, s)
    inputs_dev = {k: v.to(device) for k, v in inputs.items()}
    with torch.set_grad_enabled(True):
        tt_res = unpack_forward_output(tt_compiled(**inputs_dev))
    torch_xla.sync(wait=True)
    with torch.set_grad_enabled(True):
        tt_res.backward(gradient=rg.to(device))
    for param in model.parameters():
        if param.grad is not None and param in shard_specs:
            param.grad = sharding_constraint_tensor(param.grad, mesh, shard_specs[param])
    wanted = [p.grad for p in model.parameters() if p.grad is not None]
    torch_xla._XLAC._xla_sync_multi(wanted, list(set(p.device.type for p in wanted)), wait=True)
    tt_grads = {n: p.grad.to("cpu") for n, p in model.named_parameters() if p.grad is not None}

    # Compare
    print("[fwd] PCC=" + str(round(pcc(cpu_res.detach(), tt_res.detach().to("cpu")), 6)))
    for name in sorted(cpu_grads):
        if name not in tt_grads:
            print("[bwd] " + name + ": MISSING"); continue
        c, t = cpu_grads[name], tt_grads[name]
        p = pcc(c, t)
        cn, tn = c.float().norm().item(), t.float().norm().item()
        ratio = tn / max(cn, 1e-12)
        cz = int((c.abs() > 1e-8).sum())
        tz = int((t.abs() > 1e-8).sum())
        info = name + " PCC=" + str(round(p, 6))
        info += " sh=" + str(list(c.shape)) + "/" + str(list(t.shape))
        info += " norm=" + str(round(cn, 6)) + "/" + str(round(tn, 6))
        info += " ratio=" + str(round(ratio, 3))
        info += " nnz=" + str(cz) + "/" + str(tz)
        print("[bwd] " + info)
        if "experts" in name and c.dim() == 3 and c.shape[0] == t.shape[0]:
            pe = [pcc(c[e], t[e]) for e in range(c.shape[0])]
            good = sum(1 for x in pe if x > 0.5)
            bad = sum(1 for x in pe if x <= 0.5)
            print("  per-expert: good=" + str(good) + " bad=" + str(bad) +
                  " best=" + str(round(max(pe), 3)) + " worst=" + str(round(min(pe), 3)))
    print("Done.")

if __name__ == "__main__":
    main()
