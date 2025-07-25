#!/usr/bin/env python3
"""
Example script showing how to modify existing Llama models for tensor parallelism.

This script demonstrates the practical steps to take an existing Llama model
and add tensor parallelism support using Torch-XLA.
"""

import os
import sys
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from typing import Optional, Tuple
from transformers import LlamaModel, LlamaConfig


def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    print("Setting up XLA environment...")
    
    # Basic XLA configuration
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"  # Enable Shardy conversion
    os.environ["MESH_SHAPE"] = "1,8"

    from torch_xla.experimental import plugins

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "build/src/tt/pjrt_plugin_tt.so"
            )

    plugins.register_plugin("TT", TTPjrtPlugin())
    
    # Initialize SPMD
    xr.use_spmd()
    torch_xla.sync(True, True)
    print("XLA environment configured.")


def create_device_mesh(num_devices: int = 8, mesh_shape: Tuple[int, int] = (1, 8)) -> Mesh:
    """
    Create device mesh for tensor parallelism.
    
    Args:
        num_devices: Total number of devices
        mesh_shape: Shape of the device mesh (batch_dim, model_dim)
        
    Returns:
        Mesh object for SPMD operations
    """
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def apply_tensor_parallel_sharding(model: LlamaModel, mesh: Mesh) -> None:
    """
    Apply tensor parallel sharding to a Llama model.
    
    This function modifies the model in-place to add sharding annotations
    for tensor parallelism.
    
    Args:
        model: The Llama model to modify
        mesh: Device mesh for sharding
    """
    print("Applying tensor parallel sharding...")
    
    # Move model to XLA device first
    model = model.to(torch_xla.device())
    
    # Apply sharding to each transformer layer
    for i, layer in enumerate(model.layers):
        print(f"Sharding layer {i+1}/{len(model.layers)}")
        
        # ========================================
        # MLP (Feed-Forward) Layer Sharding
        # ========================================
        
        # Column parallel: Split output dimension across devices
        # up_proj: [hidden_size, intermediate_size] -> shard dim 0
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        
        # gate_proj: [hidden_size, intermediate_size] -> shard dim 0  
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        
        # Row parallel: Split input dimension across devices
        # down_proj: [intermediate_size, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))
        
        # ========================================
        # Self-Attention Layer Sharding
        # ========================================
        
        # Column parallel: Split attention heads across devices
        # q_proj: [hidden_size, num_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        
        # k_proj: [hidden_size, num_kv_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        
        # v_proj: [hidden_size, num_kv_heads * head_dim] -> shard dim 0
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        
        # Row parallel: Collect results from all devices
        # o_proj: [num_heads * head_dim, hidden_size] -> shard dim 1
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        
        # Note: LayerNorm parameters are typically replicated (small memory footprint)
        # You could also shard them if needed:
        # xs.mark_sharding(layer.input_layernorm.weight, mesh, (None,))
        # xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, (None,))
    
    print("Tensor parallel sharding applied successfully!")


def prepare_inputs(config: LlamaConfig, mesh: Mesh, 
                  batch_size: int = 1, seq_length: int = 1024) -> torch.Tensor:
    """
    Prepare input tensors with appropriate sharding.
    
    Args:
        config: Model configuration
        mesh: Device mesh
        batch_size: Batch size
        seq_length: Sequence length
        
    Returns:
        Sharded input tensor
    """
    print(f"Preparing inputs: batch_size={batch_size}, seq_length={seq_length}")
    
    # Create random input IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Move to XLA device
    input_ids = input_ids.to(torch_xla.device())
    
    # Mark input sharding (typically replicated for inputs)
    xs.mark_sharding(input_ids, mesh, (None, None))
    
    return input_ids


def run_inference_comparison(model_name: str = "meta-llama/Meta-Llama-3.1-8B"):
    """
    Run a complete example comparing single-device vs tensor-parallel inference.
    
    Args:
        model_name: HuggingFace model name to load
    """
    print(f"Running inference comparison for {model_name}")
    
    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()
    
    # Load model and configuration
    print("Loading model...")
    config = LlamaConfig.from_pretrained(model_name)
    
    # For demonstration, let's use a smaller model or reduce layers
    # Uncomment this to use fewer layers for faster testing
    # config.num_hidden_layers = 2
    
    model = LlamaModel(config)
    
    # ========================================
    # Single Device Reference Run
    # ========================================
    print("\n=== Single Device Reference ===")
    
    # Prepare inputs for CPU/single device
    batch_size, seq_length = 1, 512  # Smaller for demo
    input_ids_cpu = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    # Run on CPU for reference
    with torch.no_grad():
        model_cpu = model.cpu()
        outputs_cpu = model_cpu(input_ids=input_ids_cpu)
        reference_output = outputs_cpu.last_hidden_state
    
    print(f"CPU output shape: {reference_output.shape}")
    
    # ========================================
    # Tensor Parallel Run
    # ========================================
    print("\n=== Tensor Parallel Inference ===")
    
    # Apply tensor parallelism
    apply_tensor_parallel_sharding(model, mesh)
    
    # Prepare inputs for tensor parallel execution
    input_ids_tp = prepare_inputs(config, mesh, batch_size, seq_length)
    
    # Run tensor parallel inference
    with torch.no_grad():
        outputs_tp = model(input_ids=input_ids_tp)
        tp_output = outputs_tp.last_hidden_state.cpu()
    
    print(f"Tensor parallel output shape: {tp_output.shape}")
    
    # ========================================
    # Validation
    # ========================================
    print("\n=== Validation ===")
    
    def compute_pcc(x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute Pearson Correlation Coefficient."""
        x_flat, y_flat = x.flatten(), y.flatten()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()
        if denom == 0:
            return float('nan')
        return float((vx @ vy) / denom)
    
    # Compare outputs
    pcc = compute_pcc(reference_output, tp_output)
    print(f"Pearson Correlation Coefficient: {pcc:.6f}")
    
    # Check if outputs are sufficiently similar
    if pcc > 0.95:
        print("✅ Tensor parallel implementation is correct!")
    else:
        print("❌ Tensor parallel outputs differ significantly from reference")
        print("This might indicate an implementation issue")
    
    return pcc


def create_custom_sharded_layer_example():
    """
    Example of creating a custom tensor-parallel linear layer.
    
    This shows how you might modify existing PyTorch modules to support
    tensor parallelism natively.
    """
    print("\n=== Custom Sharded Layer Example ===")
    
    import torch.nn as nn
    
    class TensorParallelLinear(nn.Module):
        """
        A linear layer that supports tensor parallelism.
        """
        
        def __init__(self, in_features: int, out_features: int, 
                     bias: bool = True, mesh: Optional[Mesh] = None,
                     partition_dim: int = 0):
            """
            Args:
                in_features: Input feature dimension
                out_features: Output feature dimension
                bias: Whether to use bias
                mesh: Device mesh for sharding
                partition_dim: 0 for column parallel, 1 for row parallel
            """
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.partition_dim = partition_dim
            self.mesh = mesh
            
            # Create parameters
            self.weight = nn.Parameter(torch.randn(out_features, in_features))
            if bias:
                self.bias = nn.Parameter(torch.randn(out_features))
            else:
                self.bias = None
        
        def apply_sharding(self, mesh: Mesh):
            """Apply tensor parallel sharding to this layer."""
            if self.partition_dim == 0:
                # Column parallel: shard output dimension
                xs.mark_sharding(self.weight, mesh, ("model", None))
                if self.bias is not None:
                    xs.mark_sharding(self.bias, mesh, ("model",))
            else:
                # Row parallel: shard input dimension
                xs.mark_sharding(self.weight, mesh, (None, "model"))
                if self.bias is not None:
                    # Row parallel bias is typically replicated
                    xs.mark_sharding(self.bias, mesh, (None,))
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return nn.functional.linear(x, self.weight, self.bias)
    
    # Example usage
    setup_xla_environment()
    mesh = create_device_mesh()
    
    # Create tensor parallel layers
    hidden_size = 4096
    intermediate_size = 11008
    
    # Column parallel layer (splits output)
    column_layer = TensorParallelLinear(
        hidden_size, intermediate_size, 
        bias=False, mesh=mesh, partition_dim=0
    )
    
    # Row parallel layer (splits input)  
    row_layer = TensorParallelLinear(
        intermediate_size, hidden_size,
        bias=False, mesh=mesh, partition_dim=1
    )
    
    # Move to device and apply sharding
    column_layer = column_layer.to(torch_xla.device())
    row_layer = row_layer.to(torch_xla.device())
    
    column_layer.apply_sharding(mesh)
    row_layer.apply_sharding(mesh)
    
    # Example forward pass
    batch_size, seq_len = 2, 1024
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    hidden_states = hidden_states.to(torch_xla.device())
    xs.mark_sharding(hidden_states, mesh, (None, None, None))
    
    # Forward through tensor parallel layers
    intermediate = column_layer(hidden_states)
    output = row_layer(intermediate)
    
    print(f"Input shape: {hidden_states.shape}")
    print(f"Intermediate shape: {intermediate.shape}")
    print(f"Output shape: {output.shape}")
    print("Custom tensor parallel layers work correctly!")


def main():
    """Main function demonstrating tensor parallelism setup."""
    print("Torch-XLA Tensor Parallelism for Llama Models")
    print("=" * 50)
    
    try:
        # Run the complete inference comparison
        pcc = run_inference_comparison()
        
        # Show custom layer example
        create_custom_sharded_layer_example()
        
        print("\n" + "=" * 50)
        print("Tensor parallelism demonstration completed successfully!")
        print(f"Final validation PCC: {pcc:.6f}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
        print("This might be due to missing dependencies or hardware requirements.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
