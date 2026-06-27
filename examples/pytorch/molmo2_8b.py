# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Molmo2-8B text generation example.

Molmo2 (``allenai/Molmo2-8B``) is a multimodal image-text-to-text model whose
text decoder is a Qwen3-based decoder-only transformer. This example drives the
text path of ``Molmo2ForConditionalGeneration`` (``pixel_values=None``) as a
real causal-LM: it greedily generates a continuation for a text prompt on a
single TT device.

The model and tokenizer come from the tt-forge-models ``ModelLoader``. The loader
pads the prompt to a fixed static length (``real_len + max_length``) and reports
the real prompt length via ``loader.seq_len``; we generate by writing each new
token into the padding slot in place, so the device graph keeps a static shape
across decode steps (no recompilation) without a KV cache.
"""

from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.molmo2.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Number of tokens to greedily generate. Each step is a full static-shape forward
# (the loader runs the decoder with use_cache=False), so this is kept modest to
# stay within the example's device-time budget; the model configuration itself is
# unchanged (full weights, real seq length).
MAX_NEW_TOKENS = 16


# --------------------------------
# Molmo2-8B greedy generation
# --------------------------------
def molmo2_8b():
    """Greedily generate a continuation for the loader's sample prompt on TT."""
    device = torch_xla.device()

    # Build model + inputs via the tt-forge-models loader API.
    loader = ModelLoader(ModelVariant.MOLMO2_8B)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(batch_size=1)
    tokenizer = loader.tokenizer

    # The loader right-pads to a static length and records the real prompt length.
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    prompt_len = loader.seq_len
    total_len = input_ids.shape[1]
    eos_token_id = tokenizer.eos_token_id

    # Move model and inputs to the device.
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    compiled_model = torch.compile(model, backend="tt")

    generated_ids: List[int] = []
    cur_len = prompt_len
    prefill_logits = None
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            if cur_len >= total_len:
                break  # ran out of static padding budget
            output = compiled_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            logits = output.logits.to("cpu")
            if step == 0:
                prefill_logits = logits[0, cur_len - 1]
            # Next-token logits live at the last real position.
            next_token_id = int(logits[0, cur_len - 1].argmax(dim=-1))
            generated_ids.append(next_token_id)
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} -> "
                f"{tokenizer.decode([next_token_id])!r}",
                flush=True,
            )
            if next_token_id == eos_token_id:
                break
            # Write the new token into the padding slot and extend the mask
            # in place, keeping the tensor shape static across steps.
            input_ids[0, cur_len] = next_token_id
            attention_mask[0, cur_len] = 1
            cur_len += 1

    prompt_text = tokenizer.decode(
        input_ids[0, :prompt_len].cpu().tolist(), skip_special_tokens=True
    )
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return {
        "prompt": prompt_text,
        "generated_text": generated_text,
        "generated_ids": generated_ids,
        "prefill_logits": prefill_logits,
    }


def post_process_output(result):
    """Print the prompt and the generated continuation."""
    print("=" * 80)
    print("PROMPT:")
    print(result["prompt"])
    print("-" * 80)
    print("GENERATED:")
    print(result["generated_text"])
    print("=" * 80)
    print(f"({len(result['generated_ids'])} tokens generated)")


def test_molmo2_8b():
    """Smoke-test Molmo2-8B greedy generation on TT.

    Guards that the decoder produces finite next-token logits and a non-empty,
    in-vocabulary continuation on device.
    """
    xr.set_device_type("TT")

    result = molmo2_8b()

    prefill_logits = result["prefill_logits"]
    assert prefill_logits is not None, "no forward pass executed"
    assert torch.isfinite(prefill_logits.float()).all(), "non-finite prefill logits"

    generated_ids = result["generated_ids"]
    assert len(generated_ids) > 0, "no tokens were generated"
    vocab_size = prefill_logits.shape[-1]
    assert all(
        0 <= tok < vocab_size for tok in generated_ids
    ), "generated token id out of vocabulary range"
    assert result["generated_text"].strip() != "", "generated text is empty"

    print("Molmo2-8B generation smoke test passed.")


if __name__ == "__main__":
    xr.set_device_type("TT")
    result = molmo2_8b()
    post_process_output(result)
