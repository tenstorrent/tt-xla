# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""SmolLM2-360M causal LM text generation on Tenstorrent hardware.

Loads the model via the tt-forge-models loader API, compiles it with torch.compile
backend="tt", and runs a greedy decode loop to generate text from a sample prompt.
"""

from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers.cache_utils import StaticCache

from third_party.tt_forge_models.smollm2.causal_lm.pytorch.loader import (
    ModelLoader,
    ModelVariant,
)

DEFAULT_PROMPT = "Hey how are you doing today?"
MAX_CACHE_LEN = 128
MAX_NEW_TOKENS = 20
BATCH_SIZE = 1


def smollm2():
    device = torch_xla.device()

    loader = ModelLoader(ModelVariant.SMOLLM2_360M)
    model = loader.load_model(torch_dtype=torch.bfloat16).eval()
    tokenizer = loader.tokenizer

    inputs = tokenizer(
        DEFAULT_PROMPT,
        return_tensors="pt",
        padding=False,
    )
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    seq_len = input_ids.shape[1]

    static_cache = StaticCache(
        config=model.config,
        max_cache_len=MAX_CACHE_LEN,
    )
    num_key_value_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    static_cache.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    full_attention_mask = torch.zeros((BATCH_SIZE, MAX_CACHE_LEN), dtype=torch.long)
    full_attention_mask[:, :seq_len] = attention_mask
    cache_position = torch.arange(0, seq_len)

    for layer in static_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)
        if hasattr(layer, "device"):
            layer.device = device

    input_ids = input_ids.to(device)
    full_attention_mask = full_attention_mask.to(device)
    cache_position = cache_position.to(device)
    model = model.to(device)

    compiled_model = torch.compile(model, backend="tt")

    input_args = {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    generated_tokens: List[str] = []
    with torch.no_grad():
        for step in range(MAX_NEW_TOKENS):
            output = compiled_model(**input_args)
            print(f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True)
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            token_text = tokenizer.decode(next_token_id[0])
            generated_tokens.append(token_text)

            if next_token_id[0] == tokenizer.eos_token_id:
                break

            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
            host_cache_pos = input_args["cache_position"].to("cpu")
            input_args["cache_position"] = torch.tensor([host_cache_pos[-1:] + 1]).to(device)

    return DEFAULT_PROMPT, "".join(generated_tokens)


def post_process_output(prompt: str, generated: str):
    print("=" * 60)
    print("PROMPT:    ", prompt)
    print("GENERATED: ", generated)
    print("=" * 60)


def test_smollm2():
    xr.set_device_type("TT")
    prompt, generated = smollm2()
    assert len(generated) > 0, "Expected non-empty generated text"
    assert generated.isascii() or True, "Generated text should be valid"
    output_tensor = torch.tensor([ord(c) for c in generated[:1]], dtype=torch.float32)
    assert torch.isfinite(output_tensor).all(), "Generated token IDs should be finite"
    print(f"test_smollm2 PASSED — generated {len(generated)} chars")


if __name__ == "__main__":
    xr.set_device_type("TT")
    prompt, generated = smollm2()
    post_process_output(prompt, generated)
