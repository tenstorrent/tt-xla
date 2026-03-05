"""
Tiny GPT-OSS: real model classes with reduced dims for backward debugging.

Uses AutoModelForCausalLM.from_config() with overridden dimensions.
No pretrained weights needed - uses transformers default init + forced routing.

Architecture is EXACTLY GPT-OSS (same classes, same forward methods) but:
  hidden=256, inter=256, heads=8, head_dim=32, kv_heads=4
  E=32, K=4, vocab=1024, seq=32, batch=1

Run:
    source ~/tt-xla/venv/bin/activate && cd ~/tt-xla
    python3 tests/test_tiny_gpt_oss.py 2>&1 | tee out_tiny.txt
"""

import os
import sys

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from tt_torch.sparse_mlp import A2aSparseMLP
from tt_torch.sharding import sharding_constraint_tensor


def P(*a, **kw):
    print(*a, **kw, flush=True)


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

    num_devices = mesh_shape[0] * mesh_shape[1]
    dispatch_devices = mesh_shape[0]  # cluster_axis=0 → rows
    mesh = Mesh(np.arange(num_devices), mesh_shape, ("batch", "model"))
    P(f"[setup] {n} devices, mesh={mesh_shape}")

    # ---- Load GPT-OSS config with reduced dims ----
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

    # Override dimensions — keep structure, shrink sizes
    # Ratios match original: heads/model_axis=8, kv_heads/model_axis=1
    config.hidden_size = 512         # was 2880
    config.intermediate_size = 512   # was 2880
    config.num_attention_heads = 16  # was 64 (16/model_axis=2 heads per device)
    config.head_dim = 32             # was 64
    config.num_key_value_heads = 8   # was 8 (1 kv head per device on model axis)
    config.max_position_embeddings = 64  # small enough for tiny model
    config.num_hidden_layers = 1     # was 24
    config.vocab_size = 1024         # was 201088
    config.sliding_window = 32       # match seq_len
    config.use_cache = False         # training mode
    config.output_router_logits = False
    config.layer_types = ["full_attention"]  # 1 layer = full attention
    config.pad_token_id = 0                  # was 199999 (out of range for small vocab)
    config.eos_token_id = 1
    # Keep: num_local_experts=32, num_experts_per_tok=4

    P(f"[config] H={config.hidden_size}, inter={config.intermediate_size}, "
      f"E={config.num_local_experts}, K={config.num_experts_per_tok}, "
      f"heads={config.num_attention_heads}, head_dim={config.head_dim}, "
      f"kv_heads={config.num_key_value_heads}")

    # ---- Create model from config (random init, no pretrained weights) ----
    torch.manual_seed(42)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config)
    # Use float32 for debugging — eliminates bf16 precision as a variable
    model = model.to(torch.float32)

    # Replace MLP with A2aSparseMLP (same as loader.py does)
    for layer in model.model.layers:
        layer.mlp = A2aSparseMLP(
            layer.mlp,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_devices=num_devices,
            dispatch_devices=dispatch_devices,
            cluster_axis=0,
            config=config,
        )

    # Bypass fused remap kernel — use standard ops for sparsity construction
    # This tests whether tt.moe_expert_token_remap kernel diverges from CPU
    for layer in model.model.layers:
        layer.mlp.use_fused_remap = False

    # Smart init: force deterministic routing so same experts always chosen
    K = config.num_experts_per_tok
    for layer in model.model.layers:
        mlp = layer.mlp
        with torch.no_grad():
            mlp.router.bias.fill_(-100.0)
            for k in range(K):
                mlp.router.bias[k] = 100.0 + k * 10.0

    model.train()

    # ---- Input ----
    SEQ = 32
    BATCH = 1
    input_ids = torch.randint(0, config.vocab_size, (BATCH, SEQ))
    attention_mask = torch.ones_like(input_ids)
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}

    P(f"[setup] Input: {input_ids.shape}, model params: "
      f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # ==== CPU: forward + backward ====
    P("[cpu] forward+backward (inductor)...")
    cpu_compiled = torch.compile(model, backend="inductor")

    with torch.set_grad_enabled(True):
        cpu_out = cpu_compiled(**inputs)

    cpu_logits = cpu_out.logits
    random_grad = torch.randn(cpu_logits.shape, dtype=cpu_logits.dtype)

    with torch.set_grad_enabled(True):
        cpu_logits.backward(gradient=random_grad)

    cpu_grads = {
        n: p.grad.clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad()

    P(f"[cpu] fwd shape: {cpu_logits.shape}, grads: {len(cpu_grads)} params")

    # ==== TT: forward + backward ====
    P("[tt] compiling and running forward+backward...")
    tt_compiled = torch.compile(
        model, backend="tt",
        options={"tt_experimental_compile": False,
                 "tt_enable_torch_fx_fusion_pass": False},
    )

    device = torch_xla.device()
    model.to(device)

    # Build shard specs (matching loader.py exactly)
    shard_specs = {}
    shard_specs[model.model.embed_tokens.weight] = (None, "batch")
    shard_specs[model.model.norm.weight] = ("batch",)

    for layer in model.model.layers:
        # Attention: column-parallel Q/K/V, row-parallel O
        shard_specs[layer.self_attn.q_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.q_proj.bias] = ("model",)
        shard_specs[layer.self_attn.k_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.k_proj.bias] = ("model",)
        shard_specs[layer.self_attn.v_proj.weight] = ("model", "batch")
        shard_specs[layer.self_attn.v_proj.bias] = ("model",)
        shard_specs[layer.self_attn.o_proj.weight] = ("batch", "model")
        shard_specs[layer.self_attn.o_proj.bias] = ("batch",)
        shard_specs[layer.self_attn.sinks] = (None,)

        # Router
        shard_specs[layer.mlp.router.weight] = (None, "batch")

        # Expert weights: compound-sharded on E dim
        shard_specs[layer.mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
        shard_specs[layer.mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)
        shard_specs[layer.mlp.experts.down_proj] = (("model", "batch"), None, None)
        shard_specs[layer.mlp.experts.down_proj_bias] = (("model", "batch"), None)

        # LayerNorms
        shard_specs[layer.input_layernorm.weight] = ("batch",)
        shard_specs[layer.post_attention_layernorm.weight] = ("batch",)

    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)

    inputs_dev = {k: v.to(device) for k, v in inputs.items()}

    with torch.set_grad_enabled(True):
        tt_out = tt_compiled(**inputs_dev)

    tt_logits = tt_out.logits
    torch_xla.sync(wait=True)

    with torch.set_grad_enabled(True):
        tt_logits.backward(gradient=random_grad.to(device))

    # Mark gradient sharding (same as torch_model_tester.mark_gradient_sharding)
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

    tt_grads = {
        n: p.grad.to("cpu")
        for n, p in model.named_parameters()
        if p.grad is not None
    }

    # ==== Compare ====
    fwd_pcc = pcc(cpu_logits.detach(), tt_logits.detach().to("cpu"))
    P(f"\nForward PCC = {fwd_pcc:.6f}")

    mlp_pccs, attn_pccs, other_pccs = [], [], []
    for name in sorted(cpu_grads.keys()):
        if name not in tt_grads:
            P(f"  {name}: MISSING on TT")
            continue
        p_val = pcc(cpu_grads[name], tt_grads[name])
        cn = cpu_grads[name].norm().item()
        tn = tt_grads[name].norm().item()
        ratio = cn / tn if tn > 1e-12 else float('inf')

        if "mlp" in name:
            tag = "MLP"
            mlp_pccs.append(p_val)
        elif "self_attn" in name:
            tag = "ATTN"
            attn_pccs.append(p_val)
        else:
            tag = "OTHER"
            other_pccs.append(p_val)

        P(f"  {name}: PCC={p_val:.6f}  cpu_norm={cn:.6f}  tt_norm={tn:.6f}  "
          f"ratio={ratio:.2f}  ({tag})")

    P("")
    if mlp_pccs:
        P(f"MLP  avg PCC = {sum(mlp_pccs)/len(mlp_pccs):.6f}")
    if attn_pccs:
        P(f"ATTN avg PCC = {sum(attn_pccs)/len(attn_pccs):.6f}")
    if other_pccs:
        P(f"OTHER avg PCC = {sum(other_pccs)/len(other_pccs):.6f}")

    all_pccs = mlp_pccs + attn_pccs + other_pccs
    if all_pccs:
        P(f"ALL  avg PCC = {sum(all_pccs)/len(all_pccs):.6f}")

    P("\nDone.")


if __name__ == "__main__":
    main()
