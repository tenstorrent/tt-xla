# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def test_gpt_oss_120b():

    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    config.quantization_config["quant_method"] = "none"
    config.use_cache = False
    config.num_hidden_layers=8
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
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
        padding="max_length",
        max_length=128,
    )
    model = model.to(device)

    input["input_ids"] = input["input_ids"].to(device)
    input["attention_mask"] = input["attention_mask"].to(device)
    mark_sharding_on_inputs_and_model(model, mesh)

    # Compile model with TT backend
    print("Compiling model with torch.compile backend='tt'...")
    compiled_model = torch.compile(model, backend="tt")

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        output = compiled_model(**input)

    # Move output to CPU for inspection
    output_logits = output.logits.cpu()
    print(f"Output logits shape: {output_logits.shape}")
    print(f"Output logits: {output_logits}")

    next_token_logits = output_logits[:, -1, :]
    next_token_id = next_token_logits.argmax(dim=-1)
    decoded_text = tokenizer.decode(next_token_id[0].item(), skip_special_tokens=True)

    predicted_ids = output_logits.argmax(dim=-1)
    full_text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

    print(f"Decoded text: {decoded_text}")
    print(f"Full text: {full_text}")

    print("GPT-OSS tensor parallel test completed successfully.")
    
    # Flush logs to ensure all loguru logs are written
    import sys
    sys.stdout.flush()
    sys.stderr.flush()


def mark_sharding_on_inputs_and_model(model: torch.nn.Module, mesh: Mesh):
    """
    Apply tensor parallel sharding to GPT-OSS model layers.
    Args:
        model: GPT-OSS model instance
        mesh: Device mesh for SPMD operations
    """
    print("Applying tensor parallel sharding to GPT-OSS model...")

    # Apply tensor parallel sharding to each transformer layer
    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
    xs.mark_sharding(model.model.norm.weight, mesh, ("batch",))
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))
    for layer in model.model.layers:
        # Attention layer sharding
        # q_proj weight shape: [2880, 4096]
        # Sharded column-wise (head-parallel): [2880, 4096/num_devices]
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.q_proj.bias, mesh, ("model",))

        # k_proj weight shape: [2880, 512]
        # Sharded column-wise (head-parallel): [2880, 512/num_devices]
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.bias, mesh, ("model",))

        # v_proj weight shape: [2880, 512]
        # Sharded column-wise (head-parallel): [2880, 512/num_devices]
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.bias, mesh, ("model",))

        # o_proj weight shape: [4096, 2880]
        # Sharded row-wise: [4096/num_devices, 2880]
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))
        xs.mark_sharding(layer.self_attn.o_proj.bias, mesh, ("batch",))

        # sinks shape: [4096]
        # Local replication per device (row-wise)
        xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

        # MLP layer sharding
        # Router weight is replicated (no sharding)
        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, "batch"))

        # Shard experts across devices
        # For example: 32 experts / 8 devices = 4 experts per device
        # [num_experts, hidden_size, 2 * expert_dim]
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", "batch", None))
        # [num_experts, 2 * expert_dim]
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        # [num_experts, expert_dim, hidden_size]
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, "batch"))
        # [num_experts, hidden_size]
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", "batch"))

        xs.mark_sharding(layer.input_layernorm.weight, mesh, ("batch",))
        xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, ("batch",))

    print(f"Tensor parallel sharding applied to {len(model.model.layers)} layers")


def setup_spmd():
    """
    Initialize SPMD mode in torch_xla.
    Sets the CONVERT_SHLO_TO_SHARDY environment variable to enable
    conversion of StableHLO to Shardy dialect for improved sharding.
    """
    print("Setting up XLA SPMD environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA SPMD environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.
    Returns:
        Mesh object with shape (1, num_devices) for batch and model parallelism
    """
    num_devices = xr.global_runtime_device_count()
    # mesh_shape = (1, num_devices)
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")
    test_gpt_oss_120b()
