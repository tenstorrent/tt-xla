# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.modeling_outputs import CausalLMOutputWithPast

PROMPTS = [
    # "The capital of France is",
    "I like taking walks in the",
    # "I like taking walks in the",
    # "I like taking walks in the",
    # "I like taking walks in the",
    # "Tell me a joke",
    # "While ham sandwiches are great, I prefer",
    # "The first person to walk on the moon was",
    # "The most important branch of mathematics is",
]


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
    input_prompts: list,
    tokenizer: PreTrainedTokenizer,
    model_config,
    batch_size: int,
    max_cache_len: int,
):
    inputs = tokenizer(
        input_prompts,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",
        # padding="max_length",
        # max_length=8,
        padding_side="left",
    )

    static_cache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )

    num_key_value_heads = model_config.num_key_value_heads
    head_dim = model_config.hidden_size // model_config.num_attention_heads
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    input_args = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    #   Debug prints
    print("\n=== DEBUG: construct_inputs ===")
    print(f"Input prompts: '{input_prompts}'")
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs: {inputs.input_ids}")
    print(f"Attention mask shape: {inputs.attention_mask.shape}")
    print(f"Attention mask: {inputs.attention_mask}")
    print(f"Cache position shape: {cache_position.shape}")
    print(f"Cache position: {cache_position}")
    print(f"Actual sequence length (non-padding): {inputs.attention_mask.sum().item()}")
    print("=" * 50)
    return input_args


def transfer_to_device(model, input_args, device):
    # Move original inputs to device
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    # Move cache to device
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)

    # Finally, move model to device
    model = model.to(device)
    # model.model = model.model.to(device)
    return model, input_args


def mark_sharding_on_model_and_inputs(model, input_args, mesh):
    # Shard original inputs
    batch_size = input_args["input_ids"].shape[0]
    print(f"Batch size: {batch_size}")
    if batch_size == 1:
        print("Marking sharding for batch size 1")
        xs.mark_sharding(input_args["input_ids"], mesh, (None, None))
        xs.mark_sharding(input_args["attention_mask"], mesh, (None, None))
    else:
        print("Marking sharding for batch size > 1")
        xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))
        xs.mark_sharding(input_args["attention_mask"], mesh, ("batch", None))

    # Shard cache
    for layer in input_args["past_key_values"].layers:
        if batch_size == 1:
            xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
            xs.mark_sharding(layer.values, mesh, (None, "model", None, None))
        else:
            xs.mark_sharding(layer.keys, mesh, ("batch", "model", None, None))
            xs.mark_sharding(layer.values, mesh, ("batch", "model", None, None))

    # Shard model
    for layer in model.model.layers:
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, ("batch", "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))

        # xs.mark_sharding(layer.input_layernorm.weight, mesh, ("batch",))
        # xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, ("batch",))
    if batch_size == 1:
        xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
        xs.mark_sharding(model.lm_head.weight, mesh, (None, "batch"))


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh,
    max_tokens_to_generate: int,
    input_prompts: list,
):
    batch_size = input_args["input_ids"].shape[0]
    output_tokens = [[] for _ in range(batch_size)]
    times = []

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("RUNNING PREFILL")

            # Run forward pass
            start_time = time.time()
            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logiits = output.logits.to("cpu")
            next_token_id = output_logiits[:, -1].argmax(dim=-1)
            output_text = [
                tokenizer.decode(next_token_id[i]) for i in range(batch_size)
            ]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])
            # Check for EOS token and early exit
            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # Add newline after generation completes
                break
            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

            end_time = time.time()
            times.append(end_time - start_time)
    print()
    for i in range(batch_size):
        print(f"Result for batch {i}: {input_prompts[i]}{''.join(output_tokens[i])}")
        print()
    print(f"Number of tokens generated: {max_tokens_to_generate}")
    print(f"Total time: {sum(times):.2f}")
    print(f"Time for prefill: {times[0]:.2f}")
    print(f"Total time for decode: {sum(times[1:]):.2f}")
    print(f"Times for decode: {[f'{t:.2f}' for t in times[1:]]}")
    print(f"Average time for decode: {sum(times[1:]) / len(times[1:]):.2f}")


def run_llama_70b():
    # Set up config variables.
    batch_size: int = len(PROMPTS)
    max_cache_len: int = 32
    input_prompts: list = PROMPTS
    model_name: str = "meta-llama/Meta-Llama-3.1-70B"

    num_devices = xr.global_runtime_device_count()
    assert num_devices == 32, "This example needs a Galaxy (32 devices)"

    setup_spmd()

    device: torch.device = torch_xla.device()
    mesh = create_device_mesh()

    model, tokenizer = setup_model_and_tokenizer(model_name)

    input_args = construct_inputs(
        input_prompts, tokenizer, model.config, batch_size, max_cache_len
    )

    max_tokens_to_generate = max_cache_len - input_args["input_ids"].shape[1]
    # max_tokens_to_generate = 2

    model, input_args = transfer_to_device(model, input_args, device)

    mark_sharding_on_model_and_inputs(model, input_args, mesh)

    compiled_model = torch.compile(model, backend="tt")

    run_generate(
        compiled_model,
        input_args,
        tokenizer,
        device,
        mesh,
        max_tokens_to_generate,
        input_prompts,
    )


if __name__ == "__main__":
    xr.set_device_type("TT")

    run_llama_70b()
