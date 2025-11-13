import os

import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs

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
    max_cache_len: int,
):
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_cache_len,
        padding="max_length",
    )
    return inputs

def transfer_to_device(model, input_args, device):
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)
    return model, input_args

def mark_sharding_on_model(model, mesh):
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))

def run_llama_70b():
    # Set up config variables.
    batch_size: int = 1
    max_cache_len: int = 128
    input_prompt: str = "I like taking walks in the"
    model_name: str = "meta-llama/Meta-Llama-3.1-70B"

    num_devices = xr.global_runtime_device_count()
    assert num_devices == 32, "This example needs a Galaxy (32 devices)"

    setup_spmd()

    device: torch.device = torch_xla.device()
    mesh = create_device_mesh()

    model, tokenizer = setup_model_and_tokenizer(model_name)

    input_args = construct_inputs(input_prompt, tokenizer, max_cache_len)

    model, input_args = transfer_to_device(model, input_args, device)

    mark_sharding_on_model(model, mesh)

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

    run_llama_70b()