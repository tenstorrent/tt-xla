# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Standalone script for testing Kimi K2 (DeepSeek V3) attention decode without test infrastructure.
This script follows the pattern from examples/pytorch/llama.py for transparency.
"""

import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
import torch_xla.distributed.spmd as xs
from tests.utils import failed_ttmlir_compilation

from .configuration_deepseek import DeepseekV3Config
from .modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3ForCausalLM,
)
from .utils import MLACache



def setup_environment():
    """Setup XLA environment for multi-device execution."""
    print("=" * 80)
    print("Setting up XLA environment...")
    print("=" * 80)

    # Set device type to TT
    xr.set_device_type("TT")

    # Enable SPMD mode
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    print("XLA environment configured.")
    print()


def create_device_mesh():
    """Create device mesh for tensor parallelism."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    print(f"Mesh: {mesh}")
    print()
    return mesh


def setup_model_and_inputs():
    """Setup model configuration and create attention layer with inputs."""
    print("=" * 80)
    print("Setting up model and inputs...")
    print("=" * 80)

    # Load full Kimi K2 config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "config.json")

    config = DeepseekV3Config.from_json_file(config_path)
    config.num_hidden_layers = 1

    # Create attention layer
    attention = DeepseekV3Attention(config, layer_idx=0)
    attention = attention.to(torch.bfloat16)
    attention = attention.eval()

    # Setup dimensions
    max_cache_len = 1024
    batch_size = 64
    seq_len = 1  # Decode: single token

    # Create inputs
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    attention_mask = torch.rand(
        batch_size, 1, seq_len, max_cache_len, dtype=torch.bfloat16
    )
    position_ids = torch.arange(seq_len).unsqueeze(0)
    cache_positions = torch.randint(0, max_cache_len, (seq_len,), dtype=torch.long)

    # Create and initialize static cache
    static_cache = MLACache(
        config=config,
        max_cache_len=max_cache_len,
    )

    print(f"Created attention layer: {type(attention).__name__}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Num attention heads: {config.num_attention_heads}")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print(f"Max cache len: {max_cache_len}")
    print()

    # Early initialization of cache
    print("Initializing static cache...")
    static_cache.early_initialization(
        batch_size=batch_size,
        kv_lora_rank=512,
        pe_rank=64,
        dtype=torch.bfloat16,
        device="cpu"
    )

    print(f"Cache initialized with {len(static_cache.layers)} layers")
    for layer_idx, layer in enumerate(static_cache.layers):
        print(f"  Layer {layer_idx}:")
        print(f"    compressed_kv: {layer.compressed_kv.shape}, device={layer.compressed_kv.device}")
        print(f"    k_pe: {layer.k_pe.shape}, device={layer.k_pe.device}")
    print()

    return attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions


def transfer_to_device(attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions, device):
    """Transfer model and all inputs to the XLA device."""
    print("=" * 80)
    print(f"Transferring to device: {device}")
    print("=" * 80)

    # Transfer model
    print("Moving attention layer to device...")
    attention = attention.to(device)

    # Transfer cache tensors explicitly (like llama.py does)
    print("Moving cache tensors to device...")
    for layer_idx, layer in enumerate(static_cache.layers):
        print(f"  Layer {layer_idx}:")
        print(f"    compressed_kv: id(cpu)={hex(id(layer.compressed_kv))}")
        layer.compressed_kv = layer.compressed_kv.to(device)
        print(f"    compressed_kv: id(xla)={hex(id(layer.compressed_kv))}, device={layer.compressed_kv.device}")

        print(f"    k_pe: id(cpu)={hex(id(layer.k_pe))}")
        layer.k_pe = layer.k_pe.to(device)
        print(f"    k_pe: id(xla)={hex(id(layer.k_pe))}, device={layer.k_pe.device}")

        # Update aliases
        layer.keys = layer.compressed_kv
        layer.values = layer.k_pe

    # Transfer input tensors
    print("Moving input tensors to device...")
    hidden_states = hidden_states.to(device)
    print(f"  hidden_states: {hidden_states.shape}, device={hidden_states.device}")

    attention_mask = attention_mask.to(device)
    print(f"  attention_mask: {attention_mask.shape}, device={attention_mask.device}")

    position_ids = position_ids.to(device)
    print(f"  position_ids: {position_ids.shape}, device={position_ids.device}")

    cache_positions = cache_positions.to(device)
    print(f"  cache_positions: {cache_positions.shape}, device={cache_positions.device}")

    print()
    return attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions


def mark_sharding(attention, hidden_states, attention_mask, static_cache, mesh):
    """Mark sharding on model parameters, inputs, and cache tensors."""
    print("=" * 80)
    print("Marking sharding on tensors...")
    print("=" * 80)

    # Mark sharding on cache tensors
    print("Marking sharding on cache tensors...")
    for layer_idx, layer in enumerate(static_cache.layers):
        print(f"  Layer {layer_idx}:")
        print(f"    compressed_kv: id={hex(id(layer.compressed_kv))}, device={layer.compressed_kv.device}")
        xs.mark_sharding(layer.compressed_kv, mesh, ("_axis_1", None, None, None))
        print(f"      Marked with spec: ('_axis_1', None, None, None)")

        print(f"    k_pe: id={hex(id(layer.k_pe))}, device={layer.k_pe.device}")
        xs.mark_sharding(layer.k_pe, mesh, ("_axis_1", None, None, None))
        print(f"      Marked with spec: ('_axis_1', None, None, None)")

    # Mark sharding on input tensors
    print("Marking sharding on input tensors...")
    print(f"  hidden_states: {hidden_states.shape}, device={hidden_states.device}")
    xs.mark_sharding(hidden_states, mesh, ("_axis_1", None, "_axis_0"))
    print(f"    Marked with spec: ('_axis_1', None, '_axis_0')")

    print(f"  attention_mask: {attention_mask.shape}, device={attention_mask.device}")
    xs.mark_sharding(attention_mask, mesh, ("_axis_1", None, None, None))
    print(f"    Marked with spec: ('_axis_1', None, None, None)")

    # Mark sharding on model weights
    print("Marking sharding on model weights...")
    print(f"  q_b_proj.weight: {attention.q_b_proj.weight.shape}")
    xs.mark_sharding(attention.q_b_proj.weight, mesh, ("_axis_0", None))

    print(f"  kv_b_proj.weight: {attention.kv_b_proj.weight.shape}")
    xs.mark_sharding(attention.kv_b_proj.weight, mesh, ("_axis_0", None))

    print(f"  o_proj.weight: {attention.o_proj.weight.shape}")
    xs.mark_sharding(attention.o_proj.weight, mesh, (None, "_axis_0"))

    print(f"  q_a_proj.weight: {attention.q_a_proj.weight.shape}")
    xs.mark_sharding(attention.q_a_proj.weight, mesh, (None, "_axis_0"))

    print(f"  kv_a_proj_with_mqa.weight: {attention.kv_a_proj_with_mqa.weight.shape}")
    xs.mark_sharding(attention.kv_a_proj_with_mqa.weight, mesh, (None, "_axis_0"))

    print()


def run_inference(attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions):
    """Run inference on CPU and XLA device for comparison."""
    print("=" * 80)
    print("Running inference...")
    print("=" * 80)

    # CPU reference run
    # print("Running on CPU for reference...")
    # with torch.no_grad():
    #     cpu_output = attention(
    #         hidden_states.to("cpu"),
    #         attention_mask.to("cpu"),
    #         position_ids.to("cpu"),
    #         # Create a fresh cache for CPU
    #         None,
    #         False,  # output_attentions
    #         True,   # use_cache
    #         cache_positions.to("cpu"),
    #     )
    # print(f"CPU output shape: {cpu_output[0].shape}")
    # print()

    # XLA compiled run
    print("Compiling and running on XLA device...")
    compiled_attention = torch.compile(attention, backend="tt")

    with torch.no_grad():
        xla_output = compiled_attention(
            hidden_states,
            attention_mask,
            position_ids,
            static_cache,
            False,  # output_attentions
            True,   # use_cache
            cache_positions,
        )

    print(f"XLA output shape: {xla_output[0].shape}")

    # Compare outputs
    print("\nComparing outputs...")
    xla_result = xla_output[0].to("cpu").to(torch.float32)


def main():
    """Main execution function."""
    print("\n")
    print("=" * 80)
    print("Kimi K2 (DeepSeek V3) Attention Decode Test")
    print("=" * 80)
    print()

    # Setup environment
    setup_environment()

    # Create device mesh
    mesh = create_device_mesh()

    # Setup model and inputs
    attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions = setup_model_and_inputs()

    # Get XLA device
    device = torch_xla.device()
    print(f"Using XLA device: {device}")
    print()

    # Transfer to device
    attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions = transfer_to_device(
        attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions, device
    )

    # Mark sharding
    mark_sharding(attention, hidden_states, attention_mask, static_cache, mesh)

    # Run inference
    run_inference(attention, hidden_states, attention_mask, position_ids, static_cache, cache_positions)

    print("=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()

def test_kimi_k2_attention_decode():
    main()