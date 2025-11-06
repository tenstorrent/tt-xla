# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def main():
    xr.set_device_type("TT")

    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
    config.quantization_config["quant_method"] = "none"
    config.num_hidden_layers = 1
    config.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-20b",
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    input = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    device = xm.xla_device()
    torch._dynamo.reset()
    model = model.to(device)
    input["input_ids"] = input["input_ids"].to(device)
    input["attention_mask"] = input["attention_mask"].to(device)

    with torch.no_grad():
        output = model(**input)

    output.logits = output.logits.to("cpu")

    print(output.logits)


if __name__ == "__main__":
    main()
