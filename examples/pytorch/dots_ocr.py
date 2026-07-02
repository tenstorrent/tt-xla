# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
dots.ocr multimodal document-OCR example.

dots.ocr (rednote-hilab/dots.ocr) is a compact vision-language OCR model: a
NaViT-style vision transformer feeds a Qwen2 decoder-only language model. Given
a document image and the official layout+OCR prompt, it predicts a JSON
transcription of the page.

This example drives a single prefill forward on the TT device via the
tt_forge_models loader: it merges the document's vision embeddings into the
text tokens, runs the fused vision+text graph, and decodes the language model's
predicted next token(s) — i.e. the first token of the OCR transcription the
model would emit. The wrapped model exposes only a cache-free forward, so a full
autoregressive decode (which would recompile per step) is out of scope here; the
prefill forward is the faithful single-graph scenario the bringup validated.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.dots_ocr.mm_doc_ocr.pytorch import (
    ModelLoader,
    ModelVariant,
)


def run_dots_ocr():
    """Run a dots.ocr prefill forward on the TT device and return its logits."""
    # Build the model and the document-image inputs via the loader's public API.
    loader = ModelLoader(ModelVariant.DOTS_OCR)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    device = torch_xla.device()

    # Move the model and every input tensor to the device.
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compile the fused vision + text graph for the TT backend and run it.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        logits = compiled_model(**inputs)

    return logits.cpu(), loader


def post_process_output(logits, loader):
    """Print the top-5 next tokens the OCR model predicts for the document."""
    # Last-position logits give the model's first predicted output token.
    next_token_logits = logits[0, -1].float()
    probabilities = torch.softmax(next_token_logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probabilities, k=5)

    tokenizer = loader.processor.tokenizer
    predicted = tokenizer.decode([top_5_indices[0].item()])

    print("dots.ocr predicted next token for the document image:")
    print(f"  -> {predicted!r}\n")
    print("Top-5 candidate next tokens:")
    for i in range(5):
        idx = top_5_indices[i].item()
        prob = top_5_probs[i].item() * 100
        text = tokenizer.decode([idx])
        print(f"{i + 1}. {text!r} (id {idx}): {prob:.2f}%")


def test_dots_ocr():
    """Test dots.ocr prefill produces a finite, correctly shaped logit tensor."""
    xr.set_device_type("TT")

    logits, loader = run_dots_ocr()

    # Expected shape: (batch, sequence, vocab).
    vocab_size = loader.load_config().vocab_size
    assert logits.ndim == 3, f"expected 3D logits, got shape {tuple(logits.shape)}"
    assert logits.shape[0] == 1, f"expected batch 1, got {logits.shape[0]}"
    assert (
        logits.shape[-1] == vocab_size
    ), f"expected vocab dim {vocab_size}, got {logits.shape[-1]}"
    assert torch.isfinite(logits).all(), "logits contain non-finite values"

    print("dots.ocr prefill output is finite and correctly shaped.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device, so set it to the TT device.
    xr.set_device_type("TT")

    logits, loader = run_dots_ocr()
    post_process_output(logits, loader)
