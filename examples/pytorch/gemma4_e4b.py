# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "google/gemma-4-E4B-it"

DEFAULT_PROMPT = "What is your favorite city?"


def gemma4():
    device = torch_xla.device()

    # Load full multimodal model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Format with chat template
    messages = [{"role": "user", "content": DEFAULT_PROMPT}]
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer(
        [input_text],
        return_tensors="pt",
        max_length=64,
        padding="max_length",
        truncation=True,
    )

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    model = model.to(device)

    # Set compile options with bfp8 weight dtype
    torch_xla.set_custom_compile_options(
        {
            "experimental_weight_dtype": "bfp_bf8",
        }
    )

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


if __name__ == "__main__":
    xr.set_device_type("TT")
    gemma4()
