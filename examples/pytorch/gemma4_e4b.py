# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache

MODEL_NAME = "google/gemma-4-E4B-it"
DEFAULT_PROMPT = "What is your favorite city?"
MAX_LENGTH = 64
BATCH_SIZE = 1

xr.set_device_type("TT")


# ---- Tracking issue: https://github.com/tenstorrent/tt-xla/issues/TODO ----


def _load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _prepare_inputs(tokenizer):
    messages = [{"role": "user", "content": DEFAULT_PROMPT}]
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    return tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.push
def test_gemma4_prefill_no_cache():
    """Prefill pass without KV cache."""
    device = torch_xla.device()
    model, tokenizer = _load_model_and_tokenizer()
    inputs = _prepare_inputs(tokenizer)

    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp8"})
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

    logits = output.logits.to("cpu")
    next_token_id = logits[:, -1].argmax(dim=-1)
    print(f"Prompt: {DEFAULT_PROMPT}")
    print(f"Next token: {tokenizer.decode(next_token_id[0])}")


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.push
def test_gemma4_prefill_with_cache():
    """Prefill pass using StaticCache."""
    device = torch_xla.device()
    model, tokenizer = _load_model_and_tokenizer()
    inputs = _prepare_inputs(tokenizer)

    seq_len = inputs["input_ids"].shape[1]
    static_cache = StaticCache(
        config=model.config,
        max_cache_len=MAX_LENGTH,
    )
    static_cache.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=model.config.text_config.num_key_value_heads,
        head_dim=model.config.text_config.head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    cache_position = torch.arange(0, seq_len)

    for layer in static_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)

    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    cache_position = cache_position.to(device)

    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp8"})
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(
            input_ids=input_ids,
            past_key_values=static_cache,
            cache_position=cache_position,
            use_cache=True,
        )

    logits = output.logits.to("cpu")
    next_token_id = logits[:, -1].argmax(dim=-1)
    print(f"Prompt: {DEFAULT_PROMPT}")
    print(f"Next token (prefill with cache): {tokenizer.decode(next_token_id[0])}")


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.push
def test_gemma4_decode():
    """Single decode step: prefill followed by one autoregressive decode step."""
    device = torch_xla.device()
    model, tokenizer = _load_model_and_tokenizer()
    inputs = _prepare_inputs(tokenizer)

    seq_len = inputs["input_ids"].shape[1]
    # max_cache_len must accommodate both prefill tokens and at least one decode token
    static_cache = StaticCache(
        config=model.config,
        max_cache_len=seq_len + 1,
    )
    static_cache.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=model.config.text_config.num_key_value_heads,
        head_dim=model.config.text_config.head_dim,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    for layer in static_cache.layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)

    model = model.to(device)
    input_ids = inputs["input_ids"].to(device)
    cache_position = torch.arange(0, seq_len).to(device)

    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp8"})
    compiled_model = torch.compile(model, backend="tt")

    # Prefill step
    with torch.no_grad():
        prefill_output = compiled_model(
            input_ids=input_ids,
            past_key_values=static_cache,
            cache_position=cache_position,
            use_cache=True,
        )

    prefill_logits = prefill_output.logits.to("cpu")
    next_token_id = prefill_logits[:, -1].argmax(dim=-1).unsqueeze(-1)
    print(f"Prompt: {DEFAULT_PROMPT}")
    print(f"Prefill next token: {tokenizer.decode(next_token_id[0, 0])}")

    # Decode step: feed the predicted token back in
    decode_input_ids = next_token_id.to(device)
    decode_cache_position = torch.tensor([seq_len]).to(device)

    with torch.no_grad():
        decode_output = compiled_model(
            input_ids=decode_input_ids,
            past_key_values=static_cache,
            cache_position=decode_cache_position,
            use_cache=True,
        )

    decode_logits = decode_output.logits.to("cpu")
    decoded_token_id = decode_logits[:, -1].argmax(dim=-1)
    print(f"Decoded token: {tokenizer.decode(decoded_token_id[0])}")


if __name__ == "__main__":
    test_gemma4_prefill_no_cache()
