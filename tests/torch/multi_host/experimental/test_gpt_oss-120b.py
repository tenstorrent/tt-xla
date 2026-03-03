# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Multi-host tensor parallel test for GPT-OSS 120B across different topologies.

Loads model, applies tensor parallel sharding, compiles with backend 'tt',
and runs inference.

Example:
    # Run on dual_bh_quietbox
    pytest -svv tests/torch/multi_host/experimental/test_gpt_oss-120b.py -k "dual_bh_quietbox"
"""

import sys

import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd, get_mesh
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.utils.quantization_config import Mxfp4Config


def setup_spmd():
    """Enable SPMD mode with Shardy conversion (required for tensor parallel)."""
    enable_spmd()


def create_device_mesh(mesh_shape: tuple[int, int]) -> Mesh:
    """Create device mesh with specified shape and names ('batch', 'model')."""
    return get_mesh(mesh_shape, ("batch", "model"))


@pytest.mark.tensor_parallel
@pytest.mark.parametrize("topology", ["dual_bh_quietbox", "dual_galaxy", "quad_galaxy"])
def test_gpt_oss_120b(topology, configure_topology, mesh_shape):
    """
    Run GPT-OSS 120B inference with tensor parallelism.

    Loads model with quantization, applies tensor parallel sharding,
    then runs distributed inference on TT devices and decodes output.
    """
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")
    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh(mesh_shape)

    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    config.use_cache = False
    config.num_hidden_layers = 1
    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()
    
    print("Replacing MLP with A2aSparseMLP...")
    from tt_torch.sparse_mlp import A2aSparseMLP
    # "a2a_sparse": replace MLP now (used for both CPU golden and device)
    # "a2a_sparse_device_only": keep original MLP here (CPU golden uses it),
    #   replacement happens in load_shard_spec() before device compilation
    for layer in model.model.layers:
        cluster_axis = 0
        layer.mlp = A2aSparseMLP(
            layer.mlp,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_devices=mesh_shape[0] * mesh_shape[1],
            dispatch_devices=mesh_shape[cluster_axis],
            cluster_axis=cluster_axis,
            config=config,
        )
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")
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
    mark_sharding_custom(model, mesh, config)

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

    print(
        f"GPT-OSS 120B tensor parallel ({mesh_shape[0]}x{mesh_shape[1]} mesh): "
        f"Test completed successfully"
    )

    # Flush logs to ensure result is printed
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
    
def mark_sharding_custom(model, mesh, mlp_type="a2a_sparse"):
    """Apply tensor parallel sharding with support for different MLP types.

    Args:
        model: The gpt-oss model instance
        mesh: Device mesh for SPMD operations
        mlp_type: Type of MLP sharding ("standard" or "a2a_sparse")
    """
    print(f"Applying custom tensor parallel sharding (mlp_type={mlp_type})...")

    # Embedding and output layers
    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
    xs.mark_sharding(model.model.norm.weight, mesh, ("batch",))
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))

    # Apply tensor parallel sharding to each transformer layer
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
        # Router weight is replicated with batch sharding
        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, "batch"))

        if mlp_type == "a2a_sparse":
            xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, (("model", "batch"), None, None))
            xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, (("model", "batch"), None))
            xs.mark_sharding(layer.mlp.experts.down_proj, mesh, (("model", "batch"), None, None))
            xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, (("model", "batch"), None))
        else:
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

        # Layer normalization weights
        xs.mark_sharding(layer.input_layernorm.weight, mesh, ("batch",))
        xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, ("batch",))

    print(f"Custom tensor parallel sharding applied to {len(model.model.layers)} layers")

