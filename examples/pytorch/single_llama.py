# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizer,
)
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast
import os
import numpy as np
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs


# --------------------------------
# Llama Generation Example
# --------------------------------
def llama():

    # Connect the device.
    device = xm.xla_device()

    # Instantiate model.
    model_name: str = "meta-llama/Llama-3.2-3B"
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    model.config.num_hidden_layers = 28
    # Instantiate tokenizer.
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Put it in inference mode
    model = model.eval()

    # Generate inputs.
    inputs = tokenizer.encode_plus(
        "I like taking walks in the",
        return_tensors="pt",
        truncation=True,
    )
    with torch.no_grad():
        # Instantiate static cache on host then transfer it to device to avoid CE creation ops
        batch_size = 1
        max_cache_len = 32
        static_cache: StaticCache = StaticCache(
            config=model.config,
            max_batch_size=batch_size,
            max_cache_len=max_cache_len,
            device="cpu",
            # device='xla',  # 'xla' device will create the cache on host then we move it to device
            dtype=torch.bfloat16,
        )

        # move static cache to device after host-side initialization. This gets captured in the compile I think,
        # which breaks stuff up since there's technically a graph break now in the trace, as it looks like these are
        # new different inputs...

        static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
        static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]

        # Experiment - force materialization sync before compilation. FAIL
        # torch_xla.sync()
        # torch_xla._XLAC._xla_sync_multi([*static_cache.key_cache, *static_cache.value_cache], device, wait=False)

        # Experiment - remark static addresses - Fail.
        # for k, v in zip(static_cache.key_cache, static_cache.value_cache):
        #     torch._dynamo.mark_static_address(k)
        #     torch._dynamo.mark_static_address(v)

    # mark shard specs
    cache_position = torch.arange(0, inputs.input_ids.shape[1])
    input_args = {
        "input_ids": inputs.input_ids.to(device),
        "past_key_values": static_cache,
        "cache_position": cache_position.to(device),
        "use_cache": True,
    }

    # Move inputs and model to device.
    model = model.to(device)

    # Run model (with no gradient calculation since we only need inference).
    tokens_to_generate = 16

    output_tokens = []

    model.compile(backend="tt")

    with torch.no_grad():
        # Custom generation loop impl
        for step in range(tokens_to_generate):
            output: CausalLMOutputWithPast = model(**input_args)
            print(output)
            output_logits: torch.Tensor = output.logits.to("cpu")
            output_text = tokenizer.decode(output_logits[:, -1].argmax(dim=-1))

            output_tokens.append(output_text)
            print("Generated token:", output_text)

            # Update inputs for next iteration
            next_token = output_logits[:, -1].argmax(dim=-1).unsqueeze(-1)
            input_args["input_ids"] = next_token.to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

    print("output tokens:", output_tokens)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    llama()
