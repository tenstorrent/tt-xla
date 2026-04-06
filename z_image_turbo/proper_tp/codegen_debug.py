# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Fast codegen-only debug script — skips CPU reference runs.

Usage:
  python z_image_turbo/proper_tp/codegen_debug.py              # transformer only
  python z_image_turbo/proper_tp/codegen_debug.py --te         # text encoder only
  python z_image_turbo/proper_tp/codegen_debug.py --no-legacy  # without tt_legacy_compile
"""

import argparse
import os

import torch
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
from run import pad_transformer_heads, PROMPT

_HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_BASE = os.path.join(_HERE, "codegen_output")


def run_codegen(legacy_compile: bool, text_encoder_only: bool):
    patch_rope_for_tt()

    setup_spmd()
    xr.set_device_type("TT")

    num_devices = xr.global_runtime_device_count()
    print(f"Detected {num_devices} TT devices")
    assert num_devices == 4

    os.makedirs(EXPORT_BASE, exist_ok=True)
    compile_options = {"tt_legacy_compile": True} if legacy_compile else {}
    mesh = get_mesh((1, num_devices), ("batch", "model"))
    device = torch_xla.device()

    if text_encoder_only:
        tokenizer, text_encoder = load_text_encoder()
        text_encoder = text_encoder.to(device)
        apply_te_sharding_tp(text_encoder, mesh, model_axis="model")
        text_encoder._output_capturing_hooks_installed = True

        torch_xla.set_custom_compile_options({
            "optimization_level": 1,
            "backend": "codegen_py",
            "export_path": os.path.join(EXPORT_BASE, "text_encoder"),
            "export_tensors": True,
        })
        compiled_te = torch.compile(text_encoder, backend="tt", options=compile_options)

        enc = tokenizer(PROMPT, padding=False, truncation=True, max_length=128, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)

        print(f"\nRunning text encoder (tt_legacy_compile={legacy_compile})...")
        with torch.no_grad():
            te_output = compiled_te(input_ids=input_ids)
        torch_xla.sync(wait=True)
        print(f"Text encoder output shape: {te_output.last_hidden_state.shape}")
    else:
        cap_feats = [torch.randn(7, 2560, dtype=torch.bfloat16, device=device)]
        latents = make_dummy_latents(batch_size=1, height=512, width=512, device=device)
        timestep = torch.tensor([0.5], dtype=torch.bfloat16, device=device)

        transformer = load_transformer()
        pad_transformer_heads(transformer)
        transformer = transformer.to(device)
        apply_transformer_full_sharding_tp(transformer, mesh, model_axis="model")
        transformer._output_capturing_hooks_installed = True

        torch_xla.set_custom_compile_options({
            "optimization_level": 1,
            "backend": "codegen_py",
            "export_path": os.path.join(EXPORT_BASE, "transformer"),
            "export_tensors": True,
        })
        compiled_transformer = torch.compile(transformer, backend="tt", options=compile_options)

        print(f"\nRunning transformer (tt_legacy_compile={legacy_compile})...")
        with torch.no_grad():
            output = compiled_transformer(
                x=latents, t=timestep, cap_feats=cap_feats,
                patch_size=2, f_patch_size=1, return_dict=False,
            )
        torch_xla.sync(wait=True)

        out = output[0] if isinstance(output, (tuple, list)) else output
        if not isinstance(out, list):
            out = [out]
        print(f"Transformer output shapes: {[o.shape for o in out]}")

    dirs = sorted(d for d in os.listdir(EXPORT_BASE) if os.path.isdir(os.path.join(EXPORT_BASE, d)))
    print(f"Codegen outputs: {dirs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--te", action="store_true", help="Codegen text encoder only")
    parser.add_argument("--no-legacy", action="store_true", help="Use new compile instead of tt_legacy_compile")
    args = parser.parse_args()
    run_codegen(legacy_compile=not args.no_legacy, text_encoder_only=args.te)
