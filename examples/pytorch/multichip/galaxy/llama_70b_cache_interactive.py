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
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast


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
    model_config,
    batch_size: int,
    max_cache_len: int,
):
    inputs = tokenizer(
        input_prompt,
        return_tensors="pt",
        return_attention_mask=True,
        # padding="longest",
        # padding_side="left",
    )

    static_cache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    input_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }
    return input_args


def transfer_model_to_device(model, device):
    model = model.to(device)
    return model


def transfer_inputs_to_device(input_args, device):
    # Move original inputs to device
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    # Move cache to device
    input_args["past_key_values"].key_cache = [
        k.to(device) for k in input_args["past_key_values"].key_cache
    ]
    input_args["past_key_values"].value_cache = [
        v.to(device) for v in input_args["past_key_values"].value_cache
    ]
    input_args["cache_position"] = input_args["cache_position"].to(device)

    return input_args


def mark_sharding_on_model(model, mesh):
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
    xs.mark_sharding(model.lm_head.weight, mesh, (None, "batch"))


def mark_sharding_on_inputs(input_args, mesh):
    # Shard original inputs
    batch_size = input_args["input_ids"].shape[0]
    if batch_size == 1:
        xs.mark_sharding(input_args["input_ids"], mesh, (None, None))
        xs.mark_sharding(input_args["attention_mask"], mesh, (None, None))
    else:
        xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))
        xs.mark_sharding(input_args["attention_mask"], mesh, ("batch", None))

    # Shard cache
    for i, (key, value) in enumerate(
        zip(
            input_args["past_key_values"].key_cache,
            input_args["past_key_values"].value_cache,
        )
    ):
        if batch_size == 1:
            xs.mark_sharding(key, mesh, (None, "model", None, None))
            xs.mark_sharding(value, mesh, (None, "model", None, None))
        else:
            xs.mark_sharding(key, mesh, ("batch", "model", None, None))
            xs.mark_sharding(value, mesh, ("batch", "model", None, None))


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh,
    max_tokens_to_generate: int,
):
    batch_size = 1

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("Response: ", end="", flush=True)
            # Run forward pass
            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logiits = output.logits.to("cpu")
            next_token_id = output_logiits[:, -1, :].argmax(dim=-1)
            output_text = tokenizer.decode(next_token_id)
            print(output_text, end="", flush=True)

            # Check for EOS token and early exit
            if next_token_id == tokenizer.eos_token_id:
                print()  # Add newline after generation completes
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            # Reapply shardings for static cache if SPMD is enabled
            # See https://github.com/tenstorrent/tt-xla/issues/1641
            for i, (key, value) in enumerate(
                zip(
                    input_args["past_key_values"].key_cache,
                    input_args["past_key_values"].value_cache,
                )
            ):
                xs.mark_sharding(key, mesh, (None, "model", None, None))
                xs.mark_sharding(value, mesh, (None, "model", None, None))
    print()


def run_llama_70b():
    # Set up config variables.
    batch_size: int = 1
    max_cache_len: int = 100
    model_name: str = "meta-llama/Meta-Llama-3.1-70B"

    num_devices = xr.global_runtime_device_count()
    assert num_devices == 32, "This example needs a Galaxy (32 devices)"

    setup_spmd()

    device: torch.device = torch_xla.device()
    mesh = create_device_mesh()

    model, tokenizer = setup_model_and_tokenizer(model_name)
    model = transfer_model_to_device(model, device)
    mark_sharding_on_model(model, mesh)
    compiled_model = torch.compile(model, backend="tt")

    # Dry run to warm up the model
    print("Dry running the model to warm up...")
    dry_run_inputs = construct_inputs(
        "I like taking walks in the", tokenizer, model.config, batch_size, 10
    )
    dry_run_inputs = transfer_inputs_to_device(dry_run_inputs, device)
    mark_sharding_on_inputs(dry_run_inputs, mesh)
    run_generate(
        compiled_model,
        dry_run_inputs,
        tokenizer,
        device,
        mesh,
        1,
    )
    print("Model warmed up.")

    while True:
        input_prompt = input("Enter your prompt (q to quit): ")
        if input_prompt.lower() == "q":
            break

        input_args = construct_inputs(
            input_prompt, tokenizer, model.config, batch_size, max_cache_len
        )
        # max_tokens_to_generate = max_cache_len - input_args["input_ids"].shape[1]
        max_tokens_to_generate = 50

        input_args = transfer_inputs_to_device(input_args, device)
        mark_sharding_on_inputs(input_args, mesh)

        run_generate(
            compiled_model,
            input_args,
            tokenizer,
            device,
            mesh,
            max_tokens_to_generate,
        )


if __name__ == "__main__":
    xr.set_device_type("TT")

    run_llama_70b()
