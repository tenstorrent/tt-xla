# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import List

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig
from tt_torch.transformers_overrides import (
    _init_static_cache,
    override_cache_sliding_window_layers,
    override_ministral_sliding_window_causal_mask,
)

MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
DEFAULT_PROMPT = "Who would win in a fight - a dinosaur or a cow named Moo Moo?"
MAX_LENGTH = 256
BATCH_SIZE = 1


# --------------------------------
# Mistral 8B Generation Loop Example
# --------------------------------
def mistral_8b():
    # Set up config variables.
    model_name = "mistralai/Ministral-8B-Instruct-2410"

    # Connect the device
    device: torch.device = torch_xla.device()
    model, tokenizer = _load_model_and_tokenizer(model_name)

    input_args, formmatted_prompts = _prepare_inputs(
        [DEFAULT_PROMPT], tokenizer, model.config, BATCH_SIZE, MAX_LENGTH
    )
    max_tokens_to_generate = 20

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
    output_tokens: List[List[str]] = [[] for _ in range(1)]
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            output = compiled_model(**input_args)
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True
            )
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[0])]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])

            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # Add newline after generation completes
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

    print()
    for i in range(1):
        print(f"=" * 80)
        print(f"Result for user {i}:")
        print(f"-" * 80)
        print("PROMPT:")
        print(formmatted_prompts[i])
        print(f"-" * 80)
        print("GENERATED:")
        print("".join(output_tokens[i]))
        print(f"=" * 80)
        print()


def _load_model_and_tokenizer(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    override_ministral_sliding_window_causal_mask()
    model = model.eval()
    print("model", model)
    print("model.config", model.config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _prepare_inputs(
    input_prompt: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
) -> tuple[dict, List[str]]:

    formatted_prompts = []
    for prompt in input_prompt:
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        formatted_prompts.append(formatted_prompt)

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

    static_cache = StaticCache(
        config=model_config,
        max_cache_len=max_cache_len,
    )
    _init_static_cache(static_cache, model_config, BATCH_SIZE)

    sliding_window = getattr(
        model_config.get_text_config(decoder=True), "sliding_window", max_cache_len
    )
    # Override the cache sliding window layers to use the torch.compile/TT-friendly version
    override_cache_sliding_window_layers(static_cache, max_cache_len, sliding_window)

    # Attention mask is needed to ignore padding tokens in left-padded batches. The mask should match max_cache_len
    # to prevent recompilation or implicit padding by transformers, which can cause degenerate output.
    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    cache_position = torch.arange(0, seq_len)

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }
    print("input_args", input_args)
    return input_args, formatted_prompts


if __name__ == "__main__":
    xr.set_device_type("TT")
    mistral_8b()
