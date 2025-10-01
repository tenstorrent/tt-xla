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
from infra.utilities.xla_multichip_utils import enable_spmd, get_mesh
from infra.comparators.torch_comparator import TorchComparator
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
from typing import List
import pytest
from enum import Enum


class RunMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@pytest.mark.nightly
@pytest.mark.push
@pytest.mark.parametrize("run_mode", [RunMode.PREFILL, RunMode.DECODE])
def test_llama_step(run_mode):
    # Must be called at start of program.
    enable_spmd()

    # Connect the device.
    device = xm.xla_device()
    mesh = get_mesh((1, 2), ("batch", "model"))

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

    # Instantiate static cache on host then transfer it to device to avoid compiling creation ops
    batch_size = 1
    max_cache_len = 32
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )

    cache_position = torch.arange(0, inputs.input_ids.shape[1])
    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    # In decode mode, use only the first token and reset cache position
    if run_mode == RunMode.DECODE:
        input_args["input_ids"] = input_args["input_ids"][
            :, :1
        ]  # Take first token, keep batch dim
        input_args["cache_position"] = torch.tensor([0])  # Set cache position to [0]

    # CPU comparison
    cpu_output_logits: List[torch.Tensor] = []
    with torch.no_grad():
        cpu_output: CausalLMOutputWithPast = model(**input_args)
        cpu_logits = cpu_output.logits
        cpu_output_logits.append(cpu_logits)
        cpu_tok = tokenizer.decode(cpu_logits[:, -1].argmax(dim=-1))
        print("Cpu tok: ", cpu_tok)

    # Move inputs to device
    static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
    static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)

    # Mark shard specs on inputs.
    xs.mark_sharding(input_args["input_ids"], mesh, (None, None))
    xs.mark_sharding(input_args["cache_position"], mesh, (None,))

    for i, (key, value) in enumerate(
        zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        )
    ):
        xs.mark_sharding(key, mesh, (None, "model", None, None))
        xs.mark_sharding(value, mesh, (None, "model", None, None))

    # Move model to device.
    model = model.to(device)

    # Mark shard specs on model internals.
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    tokens_to_generate = 1

    output_tokens = []
    generated_output_logits: List[torch.Tensor] = []

    model.compile(backend="tt")

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
        for step in range(tokens_to_generate):
            output: CausalLMOutputWithPast = model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")
            generated_output_logits.append(output_logits)
            output_text = tokenizer.decode(output_logits[:, -1].argmax(dim=-1))

            output_tokens.append(output_text)
            print("Generated token:", output_text)

            # Update inputs for next iteration
            next_token = output_logits[:, -1].argmax(dim=-1).unsqueeze(-1)
            input_args["input_ids"] = next_token.to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            # reapply shardings for static cache (i/o inplace mutated tensors since they lose sharding annotations)
            for i, (key, value) in enumerate(
                zip(
                    input_args["past_key_values"].key_cache,
                    input_args["past_key_values"].value_cache,
                )
            ):
                xs.mark_sharding(key, mesh, (None, "model", None, None))
                xs.mark_sharding(value, mesh, (None, "model", None, None))

    comparator = TorchComparator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False),
            pcc=PccConfig(required_pcc=0.99),
        )
    )

    comparator.compare(generated_output_logits, cpu_output_logits)
    print("output tokens:", output_tokens)
