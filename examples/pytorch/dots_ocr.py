# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
dots.ocr document-OCR example (single multimodal prefill forward).

dots.ocr (rednote-hilab/dots.ocr) is a multimodal document-parsing model: a
``dots_vit`` vision tower feeding a Qwen2-style causal-LM decoder. This example
drives the model end-to-end through the tt-forge-models ``ModelLoader``: it
builds the document image + OCR prompt, runs a single multimodal prefill forward
on the Tenstorrent device via ``torch.compile(..., backend="tt")``, and decodes
the model's next-token prediction into human-readable text.

A single prefill forward is the path the model-bringup stage validated on device,
so it is the faithful scenario to demonstrate here. (A full KV-cache decode loop
over this custom ``trust_remote_code`` multimodal model is out of scope for this
example.)
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.dots_ocr.image_text_to_text.pytorch import (
    ModelLoader,
    ModelVariant,
)

DTYPE = torch.bfloat16


def dots_ocr():
    """Run a single dots.ocr multimodal prefill forward on the TT device."""
    device = torch_xla.device()

    # Build model + inputs via the tt-forge-models loader (public API only).
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=DTYPE).eval()
    inputs = loader.load_inputs(dtype_override=DTYPE)

    # Move model and every input tensor to the device.
    model = model.to(device)
    inputs = {
        k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()
    }

    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    return output, loader


def post_process_output(output, loader):
    """Decode and print the model's next-token prediction and top-5 candidates."""
    logits = output.logits.to("cpu").float()
    next_token_logits = logits[0, -1]

    # load_inputs() already populated the public ``processor`` attribute.
    tokenizer = loader.processor.tokenizer

    next_token_id = int(next_token_logits.argmax())
    next_token_text = tokenizer.decode([next_token_id])

    top5 = torch.topk(torch.softmax(next_token_logits, dim=-1), k=5)

    print("=" * 80)
    print("dots.ocr document OCR (single multimodal prefill forward)")
    print("-" * 80)
    print(f"PROMPT: {loader.sample_prompt}")
    print(f"LOGITS SHAPE: {tuple(logits.shape)}")
    print("-" * 80)
    print(f"PREDICTED NEXT TOKEN: {next_token_text!r} (id={next_token_id})")
    print("TOP-5 NEXT-TOKEN CANDIDATES:")
    for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        print(f"  {prob * 100:6.2f}%  id={idx:<7} {tokenizer.decode([idx])!r}")
    print("=" * 80)

    return next_token_text


def test_dots_ocr():
    """Guard the example: the multimodal forward produces finite, well-shaped logits."""
    xr.set_device_type("TT")

    output, loader = dots_ocr()
    logits = output.logits.to("cpu").float()

    vocab_size = loader.load_config().vocab_size
    assert logits.dim() == 3, f"expected 3D logits, got shape {tuple(logits.shape)}"
    assert (
        logits.shape[-1] == vocab_size
    ), f"expected vocab dim {vocab_size}, got {logits.shape[-1]}"
    assert torch.isfinite(logits).all(), "logits contain non-finite values"

    print("dots.ocr forward produced finite logits of the expected shape.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output, loader = dots_ocr()
    post_process_output(output, loader)
