# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
SmolLM2-360M causal language model generation example.

Demonstrates single-chip text generation on Tenstorrent hardware using the
tt-forge-models SmolLM2 loader with a static-cache prefill+decode loop.
"""

from typing import List, Tuple

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers.cache_utils import StaticCache

from third_party.tt_forge_models.smollm2.causal_lm.pytorch import ModelLoader
from third_party.tt_forge_models.smollm2.causal_lm.pytorch.loader import ModelVariant

DEFAULT_PROMPT = "The quick brown fox"
MAX_CACHE_LEN = 256
MAX_TOKENS_TO_GENERATE = 20
BATCH_SIZE = 1


def smollm2_360m() -> Tuple[List[List[str]], List[str]]:
    """Run SmolLM2-360M generation on a single TT device."""
    device = torch_xla.device()

    loader = ModelLoader(ModelVariant.SMOLLM2_360M)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.tokenizer

    input_args, formatted_prompts = _prepare_inputs(
        [DEFAULT_PROMPT], tokenizer, model.config, BATCH_SIZE, MAX_CACHE_LEN
    )

    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)
        if hasattr(layer, "device"):
            layer.device = device

    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)

    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    output_tokens: List[List[str]] = [[] for _ in range(BATCH_SIZE)]
    with torch.no_grad():
        for step in range(MAX_TOKENS_TO_GENERATE):
            output = compiled_model(**input_args)
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True
            )
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[i]) for i in range(BATCH_SIZE)]
            for i, tokens in enumerate(output_tokens):
                tokens.append(output_text[i])

            if torch.all(next_token_id == tokenizer.eos_token_id):
                break

            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
            host_cache_pos = input_args["cache_position"].to("cpu")
            input_args["cache_position"] = torch.tensor(
                [host_cache_pos[-1:] + 1]
            ).to(device)

    return output_tokens, formatted_prompts


def _prepare_inputs(
    prompts: List[str], tokenizer, config, batch_size: int, max_cache_len: int
) -> Tuple[dict, List[str]]:
    formatted_prompts = []
    for prompt in prompts:
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted = prompt
        formatted_prompts.append(formatted)

    prompt_lengths = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in formatted_prompts
    ]
    max_length = max(prompt_lengths)

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )
    seq_len = inputs.input_ids.shape[1]

    static_cache = StaticCache(config=config, max_cache_len=max_cache_len)
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=torch.bfloat16,
        device="cpu",
    )

    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :seq_len] = inputs.attention_mask

    cache_position = torch.arange(0, seq_len)

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }
    return input_args, formatted_prompts


def post_process_output(
    output_tokens: List[List[str]], formatted_prompts: List[str]
) -> None:
    """Print human-readable generation results."""
    for i in range(BATCH_SIZE):
        generated = "".join(output_tokens[i])
        print("=" * 80)
        print(f"PROMPT:    {formatted_prompts[i]}")
        print(f"GENERATED: {generated}")
        print("=" * 80)


def test_smollm2_360m():
    """Verify SmolLM2-360M produces non-empty finite output on TT device."""
    xr.set_device_type("TT")
    output_tokens, _ = smollm2_360m()
    assert len(output_tokens) == BATCH_SIZE
    generated = "".join(output_tokens[0])
    assert len(generated.strip()) > 0, "Generated text is empty"
    print(f"Generated: {generated!r}")


if __name__ == "__main__":
    xr.set_device_type("TT")
    output_tokens, formatted_prompts = smollm2_360m()
    post_process_output(output_tokens, formatted_prompts)
