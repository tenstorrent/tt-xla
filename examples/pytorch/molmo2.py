# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B text-decoder next-token prediction example.

allenai/Molmo2-8B is a custom (``trust_remote_code``) vision-language model: a
SigLIP-style vision tower plus a Qwen3-style text decoder. The full multimodal
forward contains ``.item()`` graph breaks and data-dependent adapter gathers, so
the model is brought up as separate components. This example drives the **text
decoder** component (``Molmo2TextModel`` + ``lm_head``) through the
tt-forge-models loader: it feeds a natural-language prompt, runs a single
compiled forward on a Tenstorrent device, and decodes the language model's
predicted continuation of the prompt.

The decoder wrapper is a plain ``input_ids -> logits`` forward with no KV cache,
so this is a single-forward (prefill-only) scenario — the model's top predicted
next tokens for the prompt — rather than an autoregressive decode loop.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.molmo2.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)


def molmo2():
    """Run a single Molmo2 text-decoder forward on a TT device, return logits."""
    loader = ModelLoader(ModelVariant.MOLMO2_8B)

    # bf16 weights keep the 8B decoder within a single-chip memory budget.
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(batch_size=1)

    device = torch_xla.device()
    model = model.to(device=device)
    input_ids = inputs["input_ids"].to(device=device)
    attention_mask = inputs["attention_mask"].to(device=device)

    # Quantize weights to a block-float format on device to fit the 8B decoder.
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        logits = compiled_model(input_ids, attention_mask)

    return logits.to("cpu").float()


def post_process_output(logits):
    """Print the prompt and the decoder's top-5 predicted next tokens."""
    # loader.tokenizer is populated as a side effect of load_inputs(); re-use it
    # here through the loader's public attribute to decode the predictions.
    loader = ModelLoader(ModelVariant.MOLMO2_8B)
    loader.load_inputs(batch_size=1)
    tokenizer = loader.tokenizer
    prompt = ModelLoader.sample_text

    next_token_logits = logits[0, -1]
    probabilities = torch.softmax(next_token_logits, dim=-1)
    top_5_probs, top_5_indices = torch.topk(probabilities, k=5)

    greedy_id = int(top_5_indices[0].item())
    greedy_token = tokenizer.decode([greedy_id])

    print(f'PROMPT: "{prompt}"')
    print(f'GREEDY CONTINUATION: "{prompt}{greedy_token}"')
    print("\nTop 5 predicted next tokens:")
    for i in range(5):
        idx = int(top_5_indices[i].item())
        token = tokenizer.decode([idx]).replace("\n", "\\n")
        prob = top_5_probs[i].item() * 100
        print(f"{i + 1}. {repr(token)} (id={idx}): {prob:.2f}%")


def test_molmo2():
    """Molmo2 text decoder produces finite logits and a stable greedy token."""
    xr.set_device_type("TT")

    logits = molmo2()

    # Output must be finite and shaped [batch, seq_len, vocab_size].
    assert torch.isfinite(logits).all(), "decoder logits contain NaN/Inf"
    assert logits.ndim == 3, f"expected [B, seq, vocab] logits, got {logits.shape}"
    assert logits.shape[0] == 1, f"expected batch size 1, got {logits.shape[0]}"

    # The greedy next-token prediction is deterministic for the fixed prompt.
    greedy_id = int(logits[0, -1].argmax(dim=-1).item())
    assert greedy_id == EXPECTED_GREEDY_TOKEN_ID, (
        f"expected greedy next-token id {EXPECTED_GREEDY_TOKEN_ID}, got {greedy_id}"
    )

    print(f"Molmo2 text decoder produced a stable greedy token (id={greedy_id}).")


# Greedy next-token id for ModelLoader.sample_text on the TT device (bf16
# weights): token 42780 -> " accelerate", completing "...is designed to
# accelerate".
EXPECTED_GREEDY_TOKEN_ID = 42780


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output = molmo2()
    post_process_output(output)
