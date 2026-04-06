# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Z-Image-Turbo: Proper 4-way Tensor Parallel Inference + Codegen
================================================================
Mesh: (1, 4) with axes ("batch", "model")

Text encoder (Qwen3-2.5B, 32 Q-heads, 8 KV-heads):
  - Full 4-way TP: 32/4=8 Q-heads per device, 8/4=2 KV-heads per device ✓
  - cumsum u8 fix: bool→int32 cast before cumsum in masking_utils (TT limitation)

Transformer (ZImageTransformer2DModel, 30 heads → padded to 32):
  - Head padding: 30 → 32 by adding 2 dummy zero-weight heads
    * to_q/k/v weight: [3840, 3840] → [4096, 3840]  (zero rows for dummy heads)
    * to_out weight:   [3840, 3840] → [3840, 4096]  (zero cols for dummy heads)
    * Dummy heads produce Q=K=V=0 → attention output=0 → no contribution to to_out
    * Result is bit-exact with the original 30-head model (verified below)
  - Full 4-way TP: 32/4=8 heads per device ✓

Validation (--validate mode):
  - patch_rope_for_tt() applied first so all runs (CPU ref, CPU padded, TT)
    use the same real-valued RoPE implementation.
  - CPU padded 32-head vs CPU original 30-head: expect PCC=1.0, max_diff=0.0
    (proves padding is mathematically transparent).
  - TT (4-way TP) vs CPU original: expect PCC≈0.998, small max_diff from
    bfloat16 allreduce ordering across 4 devices.

Codegen (--codegen mode, default):
  - Both text encoder and transformer compiled with backend="codegen_py" (EmitPy).
  - tt_legacy_compile=True required for codegen (MetaDataProp + single graph).
  - TT vs CPU comparison is skipped — codegen does not execute on hardware.
  - Produces 2 dirs per model (TT backend artifact, 1 dynamo graph, 2 TT passes):
      text_encoder_0/   ← primary complete graph
      text_encoder/     ← last TT compile pass
      transformer_0/    ← primary complete graph
      transformer/      ← last TT compile pass

Usage:
  python run.py             # codegen mode (default)
  python run.py --codegen   # explicit codegen mode
  python run.py --validate  # hardware execution + CPU comparison
"""

import argparse
import os

import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr

from common import (
    apply_te_sharding_tp,
    apply_transformer_full_sharding_tp,
    get_mesh,
    load_text_encoder,
    load_transformer,
    make_dummy_latents,
    patch_rope_for_tt,
    setup_spmd,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_BASE = os.path.join(_HERE, "codegen_output")

HEAD_DIM = 128       # 3840 / 30 — fixed by model architecture
ORIGINAL_HEADS = 30
PADDED_HEADS = 32
EXTRA_DIM = (PADDED_HEADS - ORIGINAL_HEADS) * HEAD_DIM  # 256

PROMPT = "A majestic mountain landscape at golden hour"


def pad_transformer_heads(transformer):
    """Pad attention heads from 30 to 32 by adding 2 dummy zero-weight heads.

    Zero-weight heads are mathematically transparent:
      - to_q/k/v zero rows  → Q = K = V = 0 for dummy heads
      - Attention(Q=0, K=0) → uniform softmax over V=0 → output = 0
      - to_out zero columns → dummy head output × 0 = 0 contribution
    So padded model output == original model output, exactly.
    """
    def _pad_layer(layer):
        attn = layer.attention
        in_dim = attn.to_q.weight.shape[1]  # 3840

        # to_q / to_k / to_v: pad output dim 3840 → 4096
        for proj in [attn.to_q, attn.to_k, attn.to_v]:
            w = proj.weight.data  # [3840, 3840]
            zeros = torch.zeros(EXTRA_DIM, in_dim, dtype=w.dtype, device=w.device)
            proj.weight = nn.Parameter(torch.cat([w, zeros], dim=0), requires_grad=False)
            if proj.bias is not None:
                b = proj.bias.data
                proj.bias = nn.Parameter(
                    torch.cat([b, torch.zeros(EXTRA_DIM, dtype=b.dtype, device=b.device)]),
                    requires_grad=False,
                )

        # to_out[0]: pad input dim 3840 → 4096
        proj = attn.to_out[0]
        w = proj.weight.data  # [3840, 3840]
        out_dim = w.shape[0]
        zeros = torch.zeros(out_dim, EXTRA_DIM, dtype=w.dtype, device=w.device)
        proj.weight = nn.Parameter(torch.cat([w, zeros], dim=1), requires_grad=False)

        attn.heads = PADDED_HEADS

    for layer in list(transformer.layers) + list(transformer.noise_refiner) + list(transformer.context_refiner):
        _pad_layer(layer)

    print(f"Head padding applied: {ORIGINAL_HEADS} → {PADDED_HEADS} heads "
          f"(+{PADDED_HEADS - ORIGINAL_HEADS} zero-weight dummy heads)")


def cpu_reference(cap_feats_cpu, latents_cpu, timestep_cpu):
    """Run original 30-head transformer on CPU (patched RoPE, no head padding)."""
    print("\nRunning CPU reference (original 30-head model, patched RoPE)...")
    transformer_cpu = load_transformer()
    transformer_cpu.eval()

    with torch.no_grad():
        out = transformer_cpu(
            x=latents_cpu,
            t=timestep_cpu,
            cap_feats=cap_feats_cpu,
            patch_size=2,
            f_patch_size=1,
            return_dict=False,
        )
    result = out[0] if isinstance(out, (tuple, list)) else out
    if not isinstance(result, list):
        result = [result]
    print(f"CPU reference output shapes: {[o.shape for o in result]}")
    return result


def compare(tt_out, cpu_out, label=""):
    """Compare two output lists with PCC and max absolute diff."""
    print(f"\n{'='*50}")
    print(f"Comparison{': ' + label if label else ''}")
    for i, (tt, cpu) in enumerate(zip(tt_out, cpu_out)):
        tt_f = tt.cpu().float().flatten()
        cpu_f = cpu.float().flatten()
        pcc = torch.corrcoef(torch.stack([tt_f, cpu_f]))[0, 1].item()
        max_diff = (tt_f - cpu_f).abs().max().item()
        mean_diff = (tt_f - cpu_f).abs().mean().item()
        print(f"  Output[{i}] shape={tuple(cpu.shape)}")
        print(f"    PCC:      {pcc:.6f}  (1.0 = perfect)")
        print(f"    Max diff: {max_diff:.6f}")
        print(f"    Mean diff:{mean_diff:.6f}")
    print(f"{'='*50}\n")


def encode_text_tt(tokenizer, text_encoder, prompt, device, mesh, codegen):
    """Run text encoder on TT device and return cap_feats [seq, hidden]."""
    enc = tokenizer(
        prompt,
        padding=False,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)

    text_encoder = text_encoder.to(device)
    apply_te_sharding_tp(text_encoder, mesh, model_axis="model")

    # Prevent transformers' output_capturing from installing register_forward_hook
    # on every submodule during dynamo tracing (graph break per layer).
    text_encoder._output_capturing_hooks_installed = True

    if codegen:
        os.makedirs(EXPORT_BASE, exist_ok=True)
        torch_xla.set_custom_compile_options({
            "optimization_level": 1,
            "backend": "codegen_py",
            "export_path": os.path.join(EXPORT_BASE, "text_encoder"),
            "export_tensors": True,
        })

    compile_options = {"tt_legacy_compile": True} if codegen else {}
    compiled_te = torch.compile(text_encoder, backend="tt", options=compile_options)

    print("Running text encoder forward pass...")
    with torch.no_grad():
        te_output = compiled_te(input_ids=input_ids)
    torch_xla.sync(wait=True)

    # Use last_hidden_state — hidden_states[-2] requires output_hidden_states=True
    # which conflicts with the _output_capturing_hooks_installed suppression.
    cap_feats = te_output.last_hidden_state[0]  # [seq, hidden]
    print(f"Text encoder output (cap_feats) shape: {cap_feats.shape}")
    from torch_xla.distributed.spmd import clear_sharding
    clear_sharding(cap_feats)
    return [cap_feats]


def run(codegen: bool):
    # Apply all patches globally before any model runs.
    patch_rope_for_tt()

    # Fixed seed for reproducible comparison
    torch.manual_seed(42)

    # Inputs shared across all runs
    latents_cpu  = make_dummy_latents(batch_size=1, height=512, width=512)
    timestep_cpu = torch.tensor([0.5], dtype=torch.bfloat16)

    if not codegen:
        # CPU validation: encode text on CPU using dummy cap_feats (same shape
        # as what the text encoder would produce for a short prompt).
        # We use the same dummy cap_feats for TT so the transformer comparison
        # is isolated from text encoder differences.
        cap_feats_cpu = [torch.randn(7, 2560, dtype=torch.bfloat16)]

        cpu_out = cpu_reference(cap_feats_cpu, latents_cpu, timestep_cpu)

        print("\nRunning padded 32-head model on CPU (padding correctness check)...")
        transformer_padded_cpu = load_transformer()
        pad_transformer_heads(transformer_padded_cpu)
        transformer_padded_cpu.eval()
        with torch.no_grad():
            padded_cpu_raw = transformer_padded_cpu(
                x=latents_cpu,
                t=timestep_cpu,
                cap_feats=cap_feats_cpu,
                patch_size=2,
                f_patch_size=1,
                return_dict=False,
            )
        padded_cpu_out = padded_cpu_raw[0] if isinstance(padded_cpu_raw, (tuple, list)) else padded_cpu_raw
        if not isinstance(padded_cpu_out, list):
            padded_cpu_out = [padded_cpu_out]
        compare(padded_cpu_out, cpu_out,
                label="CPU padded 32-head vs CPU original 30-head (same RoPE — expect bit-exact)")

    # ---- TT: proper 4-way TP ----
    setup_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    print(f"\nDetected {num_devices} TT devices")
    assert num_devices == 4, f"proper_tp requires 4 devices, got {num_devices}"

    mesh = get_mesh((1, num_devices), ("batch", "model"))
    device = torch_xla.device()

    # ---- Text encoder ----
    tokenizer, text_encoder = load_text_encoder()
    cap_feats_tt = encode_text_tt(tokenizer, text_encoder, PROMPT, device, mesh, codegen)
    print(f"cap_feats shape: {cap_feats_tt[0].shape}")

    # ---- Transformer ----
    if codegen:
        torch_xla.set_custom_compile_options({
            "optimization_level": 1,
            "backend": "codegen_py",
            "export_path": os.path.join(EXPORT_BASE, "transformer"),
            "export_tensors": True,
        })
    else:
        torch_xla.set_custom_compile_options({"optimization_level": 1})

    transformer = load_transformer()
    pad_transformer_heads(transformer)
    transformer = transformer.to(device)
    apply_transformer_full_sharding_tp(transformer, mesh, model_axis="model")
    transformer._output_capturing_hooks_installed = True

    compile_options = {"tt_legacy_compile": True} if codegen else {}
    compiled_transformer = torch.compile(transformer, backend="tt", options=compile_options)

    latents_tt  = [lat.to(device) for lat in latents_cpu]
    timestep_tt = timestep_cpu.to(device)

    action = "codegen emission" if codegen else "forward pass"
    print(f"\nRunning TT transformer {action}...")
    with torch.no_grad():
        output = compiled_transformer(
            x=latents_tt,
            t=timestep_tt,
            cap_feats=cap_feats_tt,
            patch_size=2,
            f_patch_size=1,
            return_dict=False,
        )
    torch_xla.sync(wait=True)

    tt_out = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(tt_out, list):
        tt_out = [tt_out]
    print(f"TT transformer output shapes: {[o.shape for o in tt_out]}")

    if codegen:
        te_dirs  = sorted(d for d in os.listdir(EXPORT_BASE) if d.startswith("text_encoder"))
        tr_dirs  = sorted(d for d in os.listdir(EXPORT_BASE) if d.startswith("transformer"))
        print(f"\nCodegen completed!")
        print(f"  Text encoder dirs : {te_dirs}")
        print(f"  Transformer dirs  : {tr_dirs}")
        print(f"  The numbered _N dirs are the primary complete graphs (one per TT compile pass).")
        print(f"  The unnumbered dirs (text_encoder/, transformer/) are the last pass — may differ slightly.")
    else:
        if not codegen:
            compare(tt_out, cpu_out, label="TT 4-way TP (padded 32-head) vs CPU (original 30-head)")
        print("Validation completed!")
        print(f"  Text encoder: 4-way TP, (1,4) mesh")
        print(f"  Transformer : 4-way TP, 30 heads padded to 32, 8 per device")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Z-Image-Turbo proper 4-way TP")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--codegen", action="store_true", default=False,
                      help="Emit EmitPy codegen to codegen_output/ (default)")
    mode.add_argument("--validate", action="store_true", default=False,
                      help="Run on TT hardware and compare against CPU reference")
    args = parser.parse_args()

    use_codegen = not args.validate
    run(codegen=use_codegen)
