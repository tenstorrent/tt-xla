# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
dots.ocr text-decoder generation example.

dots.ocr (``rednote-hilab/dots.ocr``) is a document-OCR VLM whose language model is a
standard Qwen2 decoder (``DotsOCRForCausalLM(Qwen2ForCausalLM)``, ~1.5B params, 28
layers, no sliding window). Its NaViT vision tower does not yet compile on device
(patch-embed Conv2d blocker), so this example drives the *text decoder* end to end:
a chat-formatted text prompt (no image tokens, no ``pixel_values``) flows through the
Qwen2 backbone in a greedy generation loop backed by a ``StaticCache``.

Mirrors ``llama.py`` (StaticCache + ``position_ids`` decode loop), but is single-chip
and sources weights/tokenizer/prompt from the tt-forge-models loader.
"""

from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from third_party.tt_forge_models.dots_ocr.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

# Static-cache geometry. The cache is preallocated once so prefill and decode each
# compile a single fixed shape (no per-step recompile on the growing sequence).
PROMPT_LEN = 32
MAX_CACHE_LEN = 128
MAX_NEW_TOKENS = 30
BATCH_SIZE = 1


def load_model_and_tokenizer():
    """Load the dots.ocr text decoder and tokenizer via the tt-forge-models loader."""
    loader = ModelLoader(ModelVariant.BASE)
    # bfloat16 weights; the loader drops the vision tower so only the Qwen2 decoder
    # is resident.
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    return loader, model


def construct_inputs(loader, config) -> dict:
    """Build a left-padded prompt + a CPU StaticCache (transferred to device later)."""
    tokenizer = loader.tokenizer

    # Same chat-templated prompt the loader exposes for bringup.
    messages = [{"role": "user", "content": loader.sample_text}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=PROMPT_LEN,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # StaticCache must be built on CPU and transferred separately (trace/fusion issue,
    # see tt-xla#1645).
    static_cache = StaticCache(
        config=config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=MAX_CACHE_LEN,
        device="cpu",
        dtype=torch.bfloat16,
    )
    head_dim = config.hidden_size // config.num_attention_heads
    static_cache.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=config.num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    seq_len = inputs.input_ids.shape[1]
    position_ids = torch.arange(0, seq_len).unsqueeze(0)

    # Attention mask spans the full cache so padding is ignored without triggering a
    # recompile or implicit padding inside transformers.
    full_attention_mask = torch.ones(
        (BATCH_SIZE, MAX_CACHE_LEN), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :seq_len] = inputs.attention_mask

    return {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }


def transfer_to_device(model, input_args: dict, device) -> tuple:
    """Move the model, inputs, and static cache to the TT device."""
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        layer.cumulative_length = layer.cumulative_length.to(device)
        layer.device = device
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["position_ids"] = input_args["position_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    return model.to(device), input_args


def run_generate(compiled_model, input_args: dict, eos_token_id, device) -> List[int]:
    """Greedy-decode up to MAX_NEW_TOKENS, returning the generated token ids."""
    generated_ids: List[int] = []
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            print(f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True)
            output: CausalLMOutputWithPast = compiled_model(**input_args)
            next_token_id = output.logits.to("cpu")[:, -1].argmax(dim=-1)
            token = next_token_id.item()
            if token == eos_token_id:
                break
            generated_ids.append(token)

            # Feed the new token back; advance the static-cache position by one.
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
            host_pos = input_args["position_ids"].to("cpu")
            input_args["position_ids"] = torch.tensor(
                [[host_pos[0, -1].item() + 1]]
            ).to(device)
    return generated_ids


def dots_ocr() -> tuple:
    """Run dots.ocr text-decoder greedy generation on the TT device."""
    device = torch_xla.device()
    loader, model = load_model_and_tokenizer()
    input_args = construct_inputs(loader, model.config)
    model, input_args = transfer_to_device(model, input_args, device)

    compiled_model = torch.compile(model, backend="tt")
    generated_ids = run_generate(
        compiled_model, input_args, loader.tokenizer.eos_token_id, device
    )
    return loader.sample_text, loader.tokenizer.decode(generated_ids), generated_ids


def post_process_output(prompt: str, generated_text: str):
    """Print the human-readable generation result."""
    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("-" * 80)
    print("GENERATED:")
    print(generated_text)
    print("=" * 80)


def test_dots_ocr():
    """Guard the example: generation runs and yields finite, in-vocabulary tokens."""
    xr.set_device_type("TT")

    prompt, generated_text, generated_ids = dots_ocr()

    # argmax over the logits always yields a valid vocab index, so a non-empty
    # generation that decodes to non-empty text is the meaningful correctness signal.
    assert len(generated_ids) > 0, "no tokens were generated"
    assert all(t >= 0 for t in generated_ids), f"invalid token id: {generated_ids}"
    assert generated_text.strip(), "generated text is empty"
    print(f"Generated {len(generated_ids)} tokens for prompt: {prompt!r}")


if __name__ == "__main__":
    # torch_xla defaults to CPU; point it at the TT device.
    xr.set_device_type("TT")

    prompt, generated_text, _ = dots_ocr()
    post_process_output(prompt, generated_text)
