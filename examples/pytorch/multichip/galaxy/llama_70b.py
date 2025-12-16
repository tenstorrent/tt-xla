# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer


def setup_spmd():
    print("Setting up SPMD...")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    print("SPMD setup complete.")


def create_device_mesh():
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (8, 4)
    device_ids = list(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))
    return mesh


def setup_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    input_prompt: str,
    tokenizer: PreTrainedTokenizer,
    batch_size: int,
    max_cache_len: int,
):
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        truncation=True,
        return_attention_mask=True,
    )

    for key in inputs:
        inputs[key] = inputs[key].repeat_interleave(batch_size, dim=0)

    input_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }
    return input_args


def transfer_to_device(model, input_args, device):
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)
    return model, input_args


def mark_sharding_on_model_and_inputs(model, input_args, mesh):
    batch_size = input_args["input_ids"].shape[0]
    if batch_size == 1:
        xs.mark_sharding(input_args["input_ids"], mesh, (None, None))
        xs.mark_sharding(input_args["attention_mask"], mesh, (None, None))
    else:
        xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))
        xs.mark_sharding(input_args["attention_mask"], mesh, ("batch", None))
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))
    if batch_size == 1:
        xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
        xs.mark_sharding(model.lm_head.weight, mesh, (None, "batch"))


def move_hidden_states_to_cpu(hidden_states: tuple[torch.Tensor, ...]):
    return tuple(h.to("cpu") for h in hidden_states)


def run_comparison():
    batch_size: int = 1
    max_cache_len: int = 128
    # input_prompt: str = "I like taking walks in the"
    input_prompt: str = "The capital of France is"
    model_name: str = "meta-llama/Meta-Llama-3.1-70B"
    model, tokenizer = setup_model_and_tokenizer(model_name)
    input_args = construct_inputs(input_prompt, tokenizer, batch_size, max_cache_len)
    input_args["output_hidden_states"] = True

    # Run on CPU first
    cpu_model = torch.compile(model, backend="inductor")
    with torch.no_grad():
        cpu_output = cpu_model(**input_args)
        cpu_hidden_states = cpu_output.hidden_states
        print("Length of CPU hidden states: ", len(cpu_hidden_states))
        cpu_output_logits = cpu_output.logits
        next_token_id = cpu_output_logits[0, -1, :].argmax(dim=-1)
        cpu_output_text = tokenizer.decode(next_token_id)
        print("Prompt: ", input_prompt)
        print("Output: ", cpu_output_text)

    # Run on TT device
    num_devices = xr.global_runtime_device_count()
    assert num_devices == 32, "This example needs a Galaxy (32 devices)"

    setup_spmd()

    device: torch.device = torch_xla.device()
    mesh = create_device_mesh()

    model, input_args = transfer_to_device(model, input_args, device)

    mark_sharding_on_model_and_inputs(model, input_args, mesh)

    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**input_args)
        tt_hidden_states = move_hidden_states_to_cpu(output.hidden_states)
        # tt_hidden_states = output.hidden_states.to("cpu")
        tt_output_logits = output.logits.to("cpu")
        next_token_id = tt_output_logits[0, -1, :].argmax(dim=-1)
        output_text = tokenizer.decode(next_token_id)
        print("Prompt: ", input_prompt)
        print("Output: ", output_text)

    def compute_pcc(x: tuple[torch.Tensor, ...], y: tuple[torch.Tensor, ...]):
        pccs = []
        for i in range(len(x)):
            x_flat, y_flat = x[i].flatten(), y[i].flatten()
            vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
            denom = vx.norm() * vy.norm()

            if denom == 0:
                return float("nan")
            else:
                pccs.append(float((vx @ vy) / denom))
        return (min(pccs), max(pccs))

    pcc_min, pcc_max = compute_pcc(cpu_hidden_states, tt_hidden_states)
    print("Min PCC: ", pcc_min)
    print("Max PCC: ", pcc_max)


def run_llama_70b():
    # Set up config variables.
    batch_size: int = 4
    max_cache_len: int = 128
    input_prompt: str = "The capital of France is"
    model_name: str = "meta-llama/Meta-Llama-3.1-70B"

    num_devices = xr.global_runtime_device_count()
    assert num_devices == 32, "This example needs a Galaxy (32 devices)"

    setup_spmd()

    device: torch.device = torch_xla.device()
    mesh = create_device_mesh()

    model, tokenizer = setup_model_and_tokenizer(model_name)

    input_args = construct_inputs(input_prompt, tokenizer, batch_size, max_cache_len)

    model, input_args = transfer_to_device(model, input_args, device)

    mark_sharding_on_model_and_inputs(model, input_args, mesh)

    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**input_args)
        output_logits = output.logits.to("cpu")
        next_token_id = output_logits[0, -1, :].argmax(dim=-1)
        output_text = tokenizer.decode(next_token_id)
        print("Prompt: ", input_prompt)
        print("Output: ", output_text)


if __name__ == "__main__":
    xr.set_device_type("TT")

    # run_llama_70b()
    run_comparison()
