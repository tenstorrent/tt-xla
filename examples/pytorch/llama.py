# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
    StaticCache,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

# --------------------------------
# Llama Generation Example
# --------------------------------
def llama():
    # Instantiate model.
    model_name:str = "meta-llama/Llama-3.2-3B"
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.config.num_hidden_layers = 28

    # Instantiate tokenizer.
    tokenizer:PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token


    # Put it in inference mode and torch.compile it.
    model = model.eval()
    model.compile(backend="tt")


    # Generate inputs.
    inputs = tokenizer.encode_plus(
        "I like taking walks in the",
        return_tensors="pt",
        truncation=True,
    )

    # Instantiate static cache
    batch_size = 1
    max_cache_len = 128
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=model.device,
        dtype=torch.bfloat16,
    )

    cache_position = torch.arange(0, inputs.input_ids.shape[1])
    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "use_cache": True,
        "cache_position": cache_position,
    }
    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device.
    input = {k: v.to(device) for k, v in input_args.items() if hasattr(v, "to")}
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        output:CausalLMOutputWithPast = model(**input)
        output_logits: torch.Tensor = output.logits.to("cpu")
        output_text = tokenizer.decode(output_logits[:, -1].argmax(dim=-1))

    print(output_text)

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()
