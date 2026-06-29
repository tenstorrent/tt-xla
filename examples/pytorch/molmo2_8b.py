# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B image captioning example.

Molmo2-8B (allenai) is a vision-language model: a SigLIP-style vision tower +
an MLP adapter + a Qwen3-8B text decoder. This example drives the model through
the tt_forge_models loader API in a realistic image-to-text scenario: given the
sample image and the prompt "Describe this image.", it greedily decodes a short
caption on a Tenstorrent device.

The model's vision-merge path (``build_batched_images``) contains a
data-dependent ``int(counts.sum().item())`` that cannot be traced by
``torch.compile`` on the device. So, mirroring how the model was brought up
(vision validated separately; decoder+lm_head validated on real image+text
embeddings), the vision tower + image/text embedding merge runs once on CPU to
produce ``inputs_embeds``, and only the Qwen3 decoder + lm_head are compiled and
run on device. Passing ``inputs_embeds`` with ``pixel_values=None`` makes the
model skip the vision path entirely.

The decode loop is KV-cache free and uses a fixed-length ``inputs_embeds`` buffer
(the prompt embeddings are right-padded to ``prompt_len + MAX_NEW_TOKENS``), so
the compiled graph keeps a constant shape and compiles exactly once. Each step
appends the greedily-chosen token's text embedding into the next buffer slot.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.molmo2.image_text_generation.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Number of caption tokens to generate. Kept modest because the cache-free loop
# re-runs the full decoder forward each step; the model configuration
# (resolution, prompt) is left untouched.
MAX_NEW_TOKENS = 24


def _build_prompt_embeddings(model, inputs):
    """Run the vision tower + image/text embedding merge once, on CPU.

    Returns the merged ``inputs_embeds`` for the prompt (shape (1, prompt_len,
    hidden)). This is the un-compilable part of the model (data-dependent image
    bookkeeping), so it is done eagerly on host before the device forward.
    """
    inner = model.model  # Molmo2Model: owns the vision tower + embedding merge
    images, token_pooling = inner.merge_visual_inputs(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        image_token_pooling=inputs["image_token_pooling"],
        image_grids=inputs["image_grids"],
        image_num_crops=inputs["image_num_crops"],
    )
    prompt_embeds, _ = inner.build_input_embeddings(
        inputs["input_ids"], images, token_pooling
    )
    return prompt_embeds


def _generate(max_new_tokens: int):
    """Greedily caption the loader's sample image on a TT device.

    Returns:
        tuple(list[int], torch.Tensor, tokenizer): the generated token ids, the
        final step's CPU logits (used by the test for a finiteness check), and
        the tokenizer for decoding.
    """
    loader = ModelLoader(ModelVariant.MOLMO2_8B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.processor.tokenizer
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    # --- Host: build the merged image+text prompt embeddings (vision path) ---
    with torch.no_grad():
        prompt_embeds = _build_prompt_embeddings(model, inputs)

    prompt_len = prompt_embeds.shape[1]
    hidden = prompt_embeds.shape[-1]
    total_len = prompt_len + max_new_tokens

    # Fixed-length host buffers; trailing slots are padding until filled.
    inputs_embeds = torch.zeros((1, total_len, hidden), dtype=prompt_embeds.dtype)
    inputs_embeds[:, :prompt_len] = prompt_embeds
    attention_mask = torch.zeros((1, total_len), dtype=inputs["attention_mask"].dtype)
    attention_mask[:, :prompt_len] = inputs["attention_mask"]
    token_type_ids = torch.zeros((1, total_len), dtype=inputs["token_type_ids"].dtype)
    token_type_ids[:, :prompt_len] = inputs["token_type_ids"]

    # --- Device: compile and run the decoder + lm_head only ---
    device = torch_xla.device()
    torch_xla.set_custom_compile_options({"optimization_level": 2})
    model = model.to(device)
    embed_tokens = model.model.get_input_embeddings()  # text embedding (wte)
    compiled_model = torch.compile(model, backend="tt")

    generated = []
    cur = prompt_len
    with torch.no_grad():
        for step in range(max_new_tokens):
            output = compiled_model(
                inputs_embeds=inputs_embeds.to(device),
                attention_mask=attention_mask.to(device),
                token_type_ids=token_type_ids.to(device),
                pixel_values=None,
            )
            logits = output.logits.to("cpu")
            next_token_id = int(logits[0, cur - 1].argmax(dim=-1))
            generated.append(next_token_id)
            print(f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True)

            if next_token_id == tokenizer.eos_token_id:
                break

            # Embed the chosen token (text only) and commit it to the next slot.
            tok = torch.tensor([[next_token_id]], device=device)
            inputs_embeds[0, cur] = embed_tokens(tok).to("cpu")[0, 0]
            attention_mask[0, cur] = 1
            cur += 1

    return generated, logits, tokenizer


def post_process_output(generated, tokenizer):
    """Decode and print the human-readable caption."""
    caption = tokenizer.decode(generated, skip_special_tokens=True).strip()
    print("=" * 80)
    print("PROMPT:")
    print("Describe this image.")
    print("-" * 80)
    print("GENERATED CAPTION:")
    print(caption)
    print("=" * 80)
    return caption


def test_molmo2_8b():
    """Single-forward correctness guard: logits are finite and a token decodes."""
    xr.set_device_type("TT")

    generated, logits, tokenizer = _generate(max_new_tokens=1)

    assert torch.isfinite(logits).all(), "Molmo2 logits contain NaN/Inf"
    assert len(generated) == 1, "expected exactly one greedy token"

    text = tokenizer.decode(generated, skip_special_tokens=True)
    assert isinstance(text, str)
    print(f"First predicted token decodes to: {text!r}")


if __name__ == "__main__":
    # torch_xla defaults to CPU; select the Tenstorrent device.
    xr.set_device_type("TT")

    generated, _, tokenizer = _generate(max_new_tokens=MAX_NEW_TOKENS)
    post_process_output(generated, tokenizer)
