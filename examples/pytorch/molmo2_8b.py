# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B text-decoder next-token-prediction example.

Molmo2-8B is a custom ``trust_remote_code`` VLM (Qwen3-8B text decoder +
SigLIP-style ViT). Its full ``Molmo2ForConditionalGeneration`` forward has
``.item()`` graph breaks and data-dependent adapter gathers, so the
tt-forge-models loader brings up the device-compilable text decoder
(``model.transformer`` + ``lm_head``) as a logits-only wrapper.

This example drives that loader on a single chip: it loads the wrapped decoder,
compiles it with the TT backend, runs one forward over a natural prompt, and
decodes the greedy next token from the ``[B, seq, vocab]`` logits — a real,
human-readable language-model result.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

# The molmo2 causal_lm package __init__ does not re-export ``ModelLoader`` (the
# loader lives only in the ``.loader`` submodule), so import it directly.
from third_party.tt_forge_models.molmo2.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)


# --------------------------------
# Molmo2-8B single-forward example
# --------------------------------
def molmo2_8b():
    """Run one Molmo2-8B text-decoder forward on a TT device and return logits."""
    # Load the wrapped text decoder + lm_head and the natural-prompt inputs via
    # the tt-forge-models loader. ``load_inputs`` also populates the public
    # ``loader.tokenizer`` (used in post-processing to decode the prediction).
    loader = ModelLoader(ModelVariant.MOLMO2_8B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs()

    device = torch_xla.device()
    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Match the bfp8 weight precision the sibling 8B decoder examples use so the
    # cold compile of this large decoder stays within budget.
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        logits = compiled_model(input_ids, attention_mask)

    return loader, logits


def post_process_output(loader, logits):
    """Decode and print the greedy next token the decoder predicts."""
    logits = logits.cpu().float()
    next_token_id = logits[0, -1].argmax(dim=-1)
    next_token = loader.tokenizer.decode(next_token_id)

    print("=" * 80)
    print("PROMPT:")
    print(loader.sample_text)
    print("-" * 80)
    print("PREDICTED NEXT TOKEN:")
    print(repr(next_token))
    print("-" * 80)
    print(f"CONTINUATION: {loader.sample_text}{next_token}")
    print("=" * 80)


def test_molmo2_8b():
    """Guard the example: finite logits of the expected vocab shape."""
    xr.set_device_type("TT")

    loader, logits = molmo2_8b()
    logits = logits.cpu().float()

    # Single batch item, one logits vector per prompt token over the vocab.
    assert logits.ndim == 3, f"expected [B, seq, vocab] logits, got {logits.shape}"
    assert logits.shape[0] == 1, f"expected batch size 1, got {logits.shape[0]}"
    assert torch.isfinite(logits).all(), "decoder produced non-finite logits"

    # Greedy next token must be a valid, in-vocab id.
    next_token_id = logits[0, -1].argmax(dim=-1).item()
    assert 0 <= next_token_id < logits.shape[-1]

    post_process_output(loader, logits)
    print("Molmo2-8B decoder produced finite, well-shaped logits.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    loader, logits = molmo2_8b()
    post_process_output(loader, logits)
