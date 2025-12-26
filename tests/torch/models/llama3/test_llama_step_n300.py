# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from enum import Enum
from typing import List

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.evaluators import TorchComparisonEvaluator
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from tests.infra.evaluators import AtolConfig, ComparisonConfig, PccConfig
from tests.infra.testers.single_chip.model.model_tester import RunMode
from tests.utils import BringupStatus, ModelGroup


class LLMRunMode(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@pytest.mark.push
@pytest.mark.dual_chip
@pytest.mark.model_test
@pytest.mark.record_test_properties(
    model_name="meta-llama/Llama-3.2-3B",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
@pytest.mark.parametrize("run_mode", [LLMRunMode.PREFILL, LLMRunMode.DECODE])
def test_llama_step(run_mode):

    # Must be called at start of program.
    xr.set_device_type("TT")
    enable_spmd()

    # Set up config variables.
    model_hidden_layers: int = 28
    batch_size: int = 1
    max_cache_len: int = 128
    input_prompt: str = "I like taking walks in the"
    model_name: str = "meta-llama/Llama-3.2-3B"

    # Connect the device and create mesh.
    device: torch.device = xm.xla_device()
    mesh: Mesh = get_mesh((1, xr.global_runtime_device_count()), ("batch", "model"))

    # Instantiate model.
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, use_cache=True
    )
    model.config.num_hidden_layers = model_hidden_layers
    model = model.eval()

    # Instantiate tokenizer.
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Generate inputs.
    inputs = tokenizer.encode_plus(
        input_prompt,
        return_tensors="pt",
        truncation=True,
    )

    # Instantiate static cache on host (device instantiation leads to trace of unfusable creation ops.)
    static_cache: StaticCache = StaticCache(
        config=model.config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )

    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])
    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    # In decode mode, use only the first token and reset cache position
    if run_mode == LLMRunMode.DECODE:
        input_args["input_ids"] = input_args["input_ids"][
            :, :1
        ]  # Take first token, keep batch dim
        input_args["cache_position"] = torch.tensor([0])  # Set cache position to [0]

    # CPU comparison for validation
    cpu_output_logits: List[torch.Tensor] = []
    with torch.no_grad():
        cpu_output: CausalLMOutputWithPast = model(**input_args)
        cpu_logits = cpu_output.logits
        cpu_output_logits.append(cpu_logits)
        cpu_tok = tokenizer.decode(cpu_logits[:, -1].argmax(dim=-1))
        print("Cpu tok: ", cpu_tok)

    # Move model and inputs to device.
    static_cache.key_cache = [k.to(device) for k in static_cache.key_cache]
    static_cache.value_cache = [v.to(device) for v in static_cache.value_cache]
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)

    model = model.to(device)

    # Mark shardings on model and inputs.
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

    # Shard model internals
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

    output_tokens: List[str] = []
    generated_output_logits: List[torch.Tensor] = []

    model.compile(backend="tt")

    # Run model (with no gradient calculation since we only need inference).
    with torch.no_grad():
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

        # Reapply shardings for static cache (i/o inplace mutated tensors since they lose sharding annotations).
        for i, (key, value) in enumerate(
            zip(
                input_args["past_key_values"].key_cache,
                input_args["past_key_values"].value_cache,
            )
        ):
            xs.mark_sharding(key, mesh, (None, "model", None, None))
            xs.mark_sharding(value, mesh, (None, "model", None, None))

    # Compare outputs for validation
    comparator = TorchComparisonEvaluator(
        ComparisonConfig(
            atol=AtolConfig(enabled=False),
            pcc=PccConfig(required_pcc=0.99),
        )
    )

    comparator.compare(generated_output_logits, cpu_output_logits)
