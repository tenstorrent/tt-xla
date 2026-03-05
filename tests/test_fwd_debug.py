"""
Narrow down forward PCC: test use_fused_remap=True vs False.
Also validates CPU A2aSparseMLP against reference dense computation.

Usage:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_fwd_debug.py 2>&1 | tee out_fwd_debug.txt
"""

import os
import sys
import torch
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


def run_tt(model, inputs, shard_specs, mesh, label):
    """Run model on TT, report forward PCC vs CPU inductor."""
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
    print(f"[{label}] Forward PCC={p:.6f}  cpu_norm={cpu_res.float().norm().item():.2f}  tt_norm={tt_cpu.float().norm().item():.2f}")
    return p


def cpu_only_pcc(model, inputs, label):
    """Compare A2aSparseMLP CPU output to a reference dense MoE computation."""
    model.eval()

    # Get hidden states at MLP input by running just attention
    # We'll hook the MLP layer to capture input and output
    captured = {}

    def hook_pre(module, args):
        captured["mlp_input"] = args[0].detach().clone()

    def hook_post(module, args, output):
        captured["mlp_output_a2a"] = output[0].detach().clone()

    layer = model.model.layers[0]
    h1 = layer.mlp.register_forward_pre_hook(hook_pre)
    h2 = layer.mlp.register_forward_hook(hook_post)

    cpu_compiled = torch.compile(model, backend="inductor")
    with torch.no_grad():
        cpu_compiled(**inputs)

    h1.remove()
    h2.remove()

    mlp_input = captured["mlp_input"]  # [B, S, H]
    mlp_output_a2a = captured["mlp_output_a2a"]  # [B, S, H]

    # Reference: dense MoE computation (no custom ops)
    mlp = layer.mlp
    B, S, H = mlp_input.shape
    K = mlp.num_experts_per_tok
    E = mlp.num_experts

    # 1. Router
    router_scores, router_indices = mlp.router(mlp_input)
    # router_scores: [B*S, E], router_indices: [B*S, K]

    # 2. Dense expert computation
    hidden_flat = mlp_input.view(B * S, H)  # [B*S, H]
    gate_up_proj = mlp.experts.gate_up_proj  # [E, H, inter*2]
    down_proj = mlp.experts.down_proj        # [E, inter, H]
    gate_up_bias = mlp.experts.gate_up_proj_bias  # [E, inter*2]
    down_bias = mlp.experts.down_proj_bias         # [E, H]
    inter = mlp.intermediate_size
    alpha = mlp.alpha
    limit = mlp.limit

    # Compute all experts for all tokens (dense reference)
    # gate_up: [B*S, H] x [E, H, inter*2] -> [B*S, E, inter*2]
    gate_up_out = torch.einsum('bh,ehn->ben', hidden_flat.float(), gate_up_proj.float())
    gate_up_out = gate_up_out + gate_up_bias.float().unsqueeze(0)

    # Activation
    gate_out = gate_up_out[..., :inter]
    up_out = gate_up_out[..., inter:]
    gate_out_c = gate_out.clamp(max=limit)
    up_out_c = up_out.clamp(-limit, limit)
    glu = gate_out_c * torch.sigmoid(gate_out_c * alpha)
    activated = (up_out_c + 1) * glu  # [B*S, E, inter]

    # Down: [B*S, E, inter] x [E, inter, H] -> [B*S, E, H]
    down_out = torch.einsum('bei,eih->beh', activated, down_proj.float())
    down_out = down_out + down_bias.float().unsqueeze(0)  # [B*S, E, H]

    # Weighted sum over top-K experts
    topk_weights = torch.gather(router_scores.float(), dim=-1, index=router_indices)  # [B*S, K]
    # Gather expert outputs for selected experts: [B*S, K, H]
    expert_outputs = torch.gather(
        down_out,
        dim=1,
        index=router_indices.unsqueeze(-1).expand(-1, -1, H)
    )  # [B*S, K, H]

    ref_output = (expert_outputs * topk_weights.unsqueeze(-1)).sum(dim=1)  # [B*S, H]
    ref_output = ref_output.view(B, S, H).to(mlp_input.dtype)

    p_a2a_vs_ref = pcc(mlp_output_a2a.float(), ref_output.float())
    print(f"[{label}] CPU A2aSparseMLP vs reference dense: PCC={p_a2a_vs_ref:.6f}  "
          f"a2a_norm={mlp_output_a2a.float().norm().item():.4f}  "
          f"ref_norm={ref_output.float().norm().item():.4f}")
    return p_a2a_vs_ref


def main():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    xr.set_device_type("TT")
    torch_xla._XLAC._init_computation_client()

    n = xr.global_runtime_device_count()
    mesh_shape = (4, 8) if n >= 32 else (2, 4)
    mesh = Mesh(np.arange(np.prod(mesh_shape)), mesh_shape, ("batch", "model"))

    print("=" * 60)
    print("Step 1: CPU validation - A2aSparseMLP vs reference dense")
    print("=" * 60)

    loader_cpu = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model_cpu = loader_cpu.load_model()
    inputs_cpu = loader_cpu.load_inputs()
    cpu_only_pcc(model_cpu, inputs_cpu, "fused_remap=True")

    print()
    print("=" * 60)
    print("Step 2: TT test - use_fused_remap=True (current)")
    print("=" * 60)

    loader1 = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model1 = loader1.load_model()
    inputs1 = loader1.load_inputs()
    shard_specs1 = loader1.load_shard_spec(model1)
    run_tt(model1, inputs1, shard_specs1, mesh, "fused_remap=True")

    print()
    print("=" * 60)
    print("Step 3: TT test - use_fused_remap=False (manual sparsity)")
    print("=" * 60)

    loader2 = ModelLoader(ModelVariant.GPT_OSS_20B, num_layers=1)
    model2 = loader2.load_model()
    inputs2 = loader2.load_inputs()
    shard_specs2 = loader2.load_shard_spec(model2)
    # Disable fused remap to use manual sparsity path
    for layer in model2.model.layers:
        layer.mlp.use_fused_remap = False
    run_tt(model2, inputs2, shard_specs2, mesh, "fused_remap=False")

    print("\nDone.")


if __name__ == "__main__":
    main()
