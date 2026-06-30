# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
dots.ocr vision-tower inference example (single TT chip).

rednote-hilab/dots.ocr is a document-OCR vision-language model: a NaViT-style
vision tower (``DotsVisionTransformer``) encodes the page image into patch
embeddings that the Qwen2 decoder then reads out as text. This example runs the
distinctive, compute-heavy front half of that pipeline - the vision tower - on a
single Tenstorrent device: it patchifies a document page, pushes it through the
42 transformer blocks + spatial-merge PatchMerger, and returns the merged
image-token embeddings the decoder consumes.

The tower's grid-derived rotary table is precomputed on the host by the loader
(``compute_vision_rotary``) and fed in as a plain tensor, so the device graph
carries no grid-dependent control flow. Mirrors ``compiler_options.py`` for the
compile/run/PCC structure and ``resnet_dp.py`` for the encoder-forward shape.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from third_party.tt_forge_models.dots_ocr.vision_tower.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


def _build_model_and_inputs():
    """Build the dots.ocr vision tower + document-image inputs via the loader."""
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    return model, inputs


def run_dots_ocr_vision_tower():
    """Encode a document image with the dots.ocr vision tower on a TT device."""
    options = {
        "optimization_level": 2,
    }
    torch_xla.set_custom_compile_options(options)

    model, inputs = _build_model_and_inputs()

    device = xm.xla_device()
    model = model.to(device)
    pixel_values = inputs["pixel_values"].to(device)
    rotary_pos_emb = inputs["rotary_pos_emb"].to(device)

    model.compile(backend="tt")

    with torch.no_grad():
        embeddings = model(pixel_values, rotary_pos_emb)

    return embeddings


def run_dots_ocr_vision_tower_cpu():
    """CPU reference forward for correctness comparison."""
    model, inputs = _build_model_and_inputs()
    with torch.no_grad():
        return model(inputs["pixel_values"], inputs["rotary_pos_emb"])


def post_process_output(embeddings):
    """Print a human-readable summary of the encoded document-image tokens."""
    emb = embeddings.cpu().float()
    num_tokens, hidden = emb.shape
    print("dots.ocr vision tower - document-image embeddings")
    print(f"  merged image tokens : {num_tokens}")
    print(f"  hidden size         : {hidden}")
    print(f"  mean / std          : {emb.mean().item():.4f} / {emb.std().item():.4f}")
    print(f"  min / max           : {emb.min().item():.4f} / {emb.max().item():.4f}")
    print("  first token (first 8 dims):")
    print("   ", [round(v, 4) for v in emb[0, :8].tolist()])
    print("These embeddings are the image features the Qwen2 decoder reads as text.")


def test_dots_ocr():
    """Vision-tower TT output matches CPU reference (finite, right shape, PCC)."""
    xr.set_device_type("TT")

    tt_output = run_dots_ocr_vision_tower()
    cpu_output = run_dots_ocr_vision_tower_cpu()

    tt_cpu = tt_output.cpu().float()
    ref = cpu_output.float()

    assert tt_cpu.shape == ref.shape, f"shape mismatch: {tt_cpu.shape} vs {ref.shape}"
    assert torch.isfinite(tt_cpu).all(), "TT output contains non-finite values"

    pcc = torch.corrcoef(torch.stack([tt_cpu.flatten(), ref.flatten()]))[0, 1].item()
    print(f"PCC: {pcc}")
    print(f"Max diff: {(tt_cpu - ref).abs().max().item()}")

    assert pcc > 0.95, f"PCC too low: {pcc}, expected > 0.95"


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    embeddings = run_dots_ocr_vision_tower()
    post_process_output(embeddings)
