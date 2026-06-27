# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B text-decoder next-token prediction example.

The tt-forge-models Molmo2 loader brings up the language-model backbone of
``allenai/Molmo2-8B`` as a single logits-only forward pass (decoder + LM head,
no KV cache). This example runs that forward on a Tenstorrent device for a
natural-language prompt and decodes the model's top-k next-token predictions,
mirroring the top-k post-processing in ``resnet_dp.py``.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.molmo2.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

MODEL_NAME = "allenai/Molmo2-8B"
TOP_K = 5


# --------------------------------
# Molmo2-8B single-forward example
# --------------------------------
def molmo2_8b():
    """Run the Molmo2 text decoder forward on a TT device.

    Returns:
        tuple: (loader, logits_on_cpu) where logits has shape
        ``[batch, seq_len, vocab_size]``.
    """
    # Load the decoder + LM head and a tokenized prompt via the loader. Loading
    # inputs also populates ``loader.tokenizer`` for decoding below.
    loader = ModelLoader(ModelVariant.MOLMO2_8B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs()

    # Move model + inputs to the Tenstorrent device and compile for it.
    device = torch_xla.device()
    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        logits = compiled_model(input_ids, attention_mask)

    return loader, logits.to("cpu")


def post_process_output(loader, logits):
    """Decode and print the top-k next-token predictions."""
    # Logits for the position that predicts the next token.
    next_token_logits = logits[0, -1].float()
    probs = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=TOP_K)

    print(f"Prompt: {loader.sample_text!r}")
    print(f"\nTop {TOP_K} next-token predictions:")
    for rank in range(TOP_K):
        tok_id = top_ids[rank].item()
        tok_text = loader.tokenizer.decode([tok_id])
        prob = top_probs[rank].item() * 100
        print(f"{rank + 1}. {tok_text!r} (id={tok_id}): {prob:.2f}%")

    return top_ids


def test_molmo2_8b():
    """Test the Molmo2 decoder produces finite logits and sensible top-k tokens."""
    xr.set_device_type("TT")

    loader, logits = molmo2_8b()

    # Full-prompt logits: [batch, seq_len, vocab].
    assert logits.ndim == 3, f"expected 3D logits, got shape {tuple(logits.shape)}"
    assert logits.shape[0] == 1, f"expected batch size 1, got {logits.shape[0]}"
    assert torch.isfinite(logits).all(), "decoder produced non-finite logits"

    top_ids = post_process_output(loader, logits)
    assert top_ids.shape[0] == TOP_K

    # The default prompt predicts " The" (token id 576) as the most likely
    # continuation; guard that stable top-1 ordering as resnet_dp.py does.
    assert top_ids[0].item() == 576, f"expected top-1 token id 576, got {top_ids[0].item()}"

    print("\nMolmo2-8B next-token prediction produced finite logits and top-k tokens.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    loader, logits = molmo2_8b()
    post_process_output(loader, logits)
