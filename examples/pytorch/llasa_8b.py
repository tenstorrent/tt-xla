# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-8B text-to-speech generation example.

Llasa is a text-to-speech system built on the LLaMA causal-LM architecture: it
appends the 65,536-token XCodec2 speech codebook to the vocabulary and then
autoregressively predicts discrete speech tokens (``<|s_*|>``) from the input
text. A separate XCodec2 vocoder (out of scope here) turns those tokens into a
waveform.

This example drives the model through the tt-forge-models ``ModelLoader`` API:
the loader builds the chat-templated TTS prompt and a fixed-length input window,
and we run a greedy decode loop on the Tenstorrent device to synthesize the
speech-token stream. The model is configured with ``use_cache=False`` by the
loader, so every step is a full forward over the same static ``[1, max_length]``
window -- a single compile is reused across all generation steps.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.llasa.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Number of speech tokens to synthesize. The loader's fixed window leaves room
# for up to (max_length - prompt_len) tokens; we cap generation here so the
# example fits the bringup step budget. Generation also stops early on the
# SPEECH_GENERATION_END token.
MAX_NEW_TOKENS = 32


# --------------------------------
# Llasa-8B TTS generation loop
# --------------------------------
def llasa_8b(max_new_tokens: int = MAX_NEW_TOKENS):
    """Synthesize speech tokens for the loader's sample text on a TT device.

    Returns:
        Tuple of (input text, decoded speech-token string, generated token ids).
    """
    # Build the model and the TTS prompt window via the loader's public API.
    loader = ModelLoader(ModelVariant.LLASA_8B)
    model = loader.load_model().eval()
    inputs = loader.load_inputs(batch_size=1)
    tokenizer = loader.tokenizer

    # input_ids/attention_mask are a fixed-length window with the prompt at the
    # front (length loader.seq_len) and a right-padded tail to grow into.
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    current_pos = loader.seq_len
    max_pos = input_ids.shape[1]
    speech_end_id = tokenizer.convert_tokens_to_ids(loader.SPEECH_GENERATION_END)

    # Connect the device, move the model over, and compile with the TT backend.
    device = torch_xla.device()
    model = model.to(device)
    compiled_model = torch.compile(model, backend="tt")

    generated_ids = []
    with torch.no_grad():
        for step in range(max_new_tokens):
            if current_pos >= max_pos:
                break

            # Every step re-runs the full uncached forward over the static
            # window, so the device tensors keep the same shape and the compiled
            # graph is reused. We feed fresh device copies and keep the
            # canonical window state on host.
            output = compiled_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            logits = output.logits.to("cpu")

            # The next speech token is predicted at the last real position.
            next_token_id = logits[:, current_pos - 1, :].argmax(dim=-1)
            token = next_token_id.item()
            print(f"[Step {step}] speech token id {token}", flush=True)

            if token == speech_end_id:
                break

            # Append the token to the window and extend the attention mask.
            input_ids[:, current_pos] = next_token_id
            attention_mask[:, current_pos] = 1
            current_pos += 1
            generated_ids.append(token)

    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return loader.sample_text, decoded, generated_ids


def post_process_output(input_text, decoded, generated_ids):
    """Print the human-readable synthesis result."""
    print("=" * 80)
    print("INPUT TEXT:")
    print(f"  {input_text}")
    print("-" * 80)
    print(f"GENERATED {len(generated_ids)} SPEECH TOKENS:")
    print(f"  {decoded}")
    print("=" * 80)
    print("(Feed these XCodec2 speech tokens to the XCodec2 vocoder to render audio.)")


def test_llasa_8b():
    """Smoke test: the device decode loop yields valid XCodec2 speech tokens."""
    xr.set_device_type("TT")

    # A couple of tokens is enough to exercise the compile + decode path.
    input_text, decoded, generated_ids = llasa_8b(max_new_tokens=2)

    assert input_text, "Loader returned an empty input text."
    assert len(generated_ids) >= 1, "No speech tokens were generated."
    # Generated ids must be valid (non-negative) vocabulary tokens, and the
    # greedy decode must produce a non-empty speech-token string.
    assert all(isinstance(t, int) and t >= 0 for t in generated_ids)
    assert decoded.strip(), "Decoded speech-token string is empty."
    print(f"Generated {len(generated_ids)} speech tokens: {decoded!r}")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device, so set it to the TT device.
    xr.set_device_type("TT")

    input_text, decoded, generated_ids = llasa_8b()
    post_process_output(input_text, decoded, generated_ids)
