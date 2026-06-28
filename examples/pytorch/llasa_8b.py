# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Llasa-8B text-to-speech token-generation example.

Llasa (HKUSTAudio/Llasa-8B) is a Llama-3.1-8B finetune whose vocabulary is
extended with XCodec2 speech tokens. Given a TTS prompt it autoregressively
emits *speech* tokens (the same way a causal LM emits text tokens); a separate
XCodec2 vocoder later turns those tokens into a waveform and is out of scope for
this example.

This script drives the model through the tt-forge-models ``ModelLoader`` API and
runs a greedy generation loop on a Tenstorrent device. To keep the compiled
graph at a single static shape (and therefore a single compile), it reuses the
preallocated, right-padded buffer that ``load_inputs`` returns: each step writes
the newly generated speech token into the next free slot of that fixed-size
buffer and re-runs the same forward. This is the exact forward graph the loader
was brought up on, looped to produce a real speech-token sequence rather than a
single tensor.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.llasa.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Number of speech tokens to generate. The loader preallocates 128 free slots
# after the prompt, so this stays well within the static buffer.
NUM_SPEECH_TOKENS = 24


def llasa_8b():
    """Run Llasa-8B greedy speech-token generation on a TT device."""
    device = torch_xla.device()

    # Load weights, tokenizer and the TTS prompt via the loader's public API.
    loader = ModelLoader(ModelVariant.LLASA_8B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs()
    tokenizer = loader.tokenizer

    # load_inputs returns a right-padded buffer of shape [1, prompt_len + 128];
    # the real prompt occupies the first `seq_len` positions and the rest is free
    # space we fill in as we decode. Keep the buffers on host as the source of
    # truth and move a copy to the device each step (shape never changes).
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = loader.seq_len

    model = model.to(device)
    compiled_model = torch.compile(model, backend="tt")

    generated_ids = []
    cur = prompt_len  # next free slot / position to predict
    eos_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        for step in range(NUM_SPEECH_TOKENS):
            output = compiled_model(
                input_ids=input_ids.to(device),
                attention_mask=attention_mask.to(device),
            )
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...",
                flush=True,
            )
            logits = output.logits.to("cpu")
            # The token for position `cur` is predicted from the last real
            # position's logits (`cur - 1`).
            next_token_id = logits[:, cur - 1].argmax(dim=-1)
            token_id = next_token_id.item()
            generated_ids.append(token_id)

            if token_id == eos_token_id:
                break

            # Append the new token to the static buffer and extend the mask.
            input_ids[:, cur] = next_token_id
            attention_mask[:, cur] = 1
            cur += 1

    prompt_text = tokenizer.decode(
        input_ids[0, :prompt_len], skip_special_tokens=False
    )
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return prompt_text, generated_ids, decoded, model.config.vocab_size


def post_process_output(prompt_text, generated_ids, decoded):
    """Print the prompt and the generated speech-token sequence."""
    print("=" * 80)
    print("PROMPT (TTS chat template):")
    print(prompt_text)
    print("-" * 80)
    print(f"GENERATED {len(generated_ids)} SPEECH TOKENS:")
    print(decoded)
    print("-" * 80)
    print(f"Raw token ids: {generated_ids}")
    print("=" * 80)
    print(
        "Note: these speech tokens feed the (out-of-scope) XCodec2 vocoder to "
        "synthesize the waveform."
    )


def test_llasa_8b():
    """Smoke test: generation runs and yields valid, finite speech tokens."""
    xr.set_device_type("TT")

    prompt_text, generated_ids, decoded, vocab_size = llasa_8b()

    # Generation produced at least one token, and every id is a valid index
    # into the (speech-extended) vocabulary.
    assert len(generated_ids) > 0, "no speech tokens were generated"
    assert all(
        0 <= tid < vocab_size for tid in generated_ids
    ), f"generated token id out of range [0, {vocab_size})"
    # Decoded output is a non-empty string.
    assert isinstance(decoded, str) and len(decoded) > 0, "empty decode"

    print(
        f"Generated {len(generated_ids)} valid speech tokens "
        f"(vocab_size={vocab_size})."
    )


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device, so set it to the TT device.
    xr.set_device_type("TT")

    prompt_text, generated_ids, decoded, _ = llasa_8b()
    post_process_output(prompt_text, generated_ids, decoded)
