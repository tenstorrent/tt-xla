# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig, LlamaModel
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from tests.torch_xla.utils import *


def apply_tensor_parallel_sharding_base(
    model: LlamaModel, mesh: Mesh, move_to_device: bool = True
) -> None:
    if move_to_device:
        model = model.to(torch_xla.device())

    # Apply sharding to each transformer layer
    for layer in model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))


def apply_tensor_parallel_sharding_causal(
    model: AutoModelForCausalLM, mesh: Mesh
) -> None:
    model = model.to(torch_xla.device())
    apply_tensor_parallel_sharding_base(model.model, mesh, move_to_device=False)
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))


@pytest.mark.parametrize(
    "run_causal",
    [True, False],
    ids=["causal", "base"],
)
@pytest.mark.parametrize("sequence_length", [128, 256, 512], ids=["128", "256", "512"])
def test_llama_8b_eager(run_causal, sequence_length):
    torch.manual_seed(42)

    setup_xla_environment_for_tp()
    mesh = create_device_mesh((1, xr.global_runtime_device_count()), ("batch", "model"))

    model_name = "meta-llama/Llama-3.1-8B"
    config = LlamaConfig.from_pretrained(model_name)
    if run_causal:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, config=config, torch_dtype=torch.bfloat16
        ).eval()
    else:
        model = LlamaModel.from_pretrained(
            model_name, config=config, torch_dtype=torch.bfloat16
        )

    batch_size = 1
    inputs = torch.randint(
        0, config.vocab_size, (batch_size, sequence_length), dtype=torch.int32
    )
    # Run model on CPU first
    outputs = model(inputs)
    if run_causal:
        cpu_outputs = outputs.logits
    else:
        cpu_outputs = outputs.last_hidden_state

    # Now run on devices
    if run_causal:
        apply_tensor_parallel_sharding_causal(model, mesh)
    else:
        apply_tensor_parallel_sharding_base(model, mesh)

    inputs = inputs.to(torch_xla.device())
    xs.mark_sharding(inputs, mesh, (None, None))  # Replicate inputs to all devices

    outputs = model(inputs)
    torch_xla.sync(True, True)  # Wait until all computations have finished
    if run_causal:
        tt_outputs = outputs.logits.to("cpu")
    else:
        tt_outputs = outputs.last_hidden_state.to("cpu")

    pcc = calculate_pcc(tt_outputs, cpu_outputs)
    print(f"PCC: {pcc}")
    assert pcc >= 0.95
