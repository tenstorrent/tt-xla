# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Torch-XLA Tensor Parallelism utilities for Llama models.

This module provides utilities and modified Llama components that support 
tensor parallelism using Torch-XLA's SPMD functionality.
"""

import os
import math
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
import numpy as np
from typing import Optional, Tuple, Union
from transformers.models.llama.modeling_llama import (
    LlamaAttention, 
    LlamaMLP, 
    LlamaDecoderLayer, 
    LlamaModel,
    LlamaConfig
)


def setup_tensor_parallel_environment(mesh_shape: Tuple[int, int] = (1, 8)):
    """
    Setup Torch-XLA tensor parallel environment.
    
    Args:
        mesh_shape: Tuple of (batch_dim, model_dim) for the device mesh
    """
    # Set environment variables for Tenstorrent plugin
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["XLA_ALWAYS_ALLREDUCE"] = "1"
    os.environ["ENABLE_AUTO_PARALLEL"] = "TRUE"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    os.environ["MESH_SHAPE"] = f"{mesh_shape[0]},{mesh_shape[1]}"
    os.environ["LOGGER_LEVEL"] = "DEBUG"
    
    # Register TT plugin
    from torch_xla.experimental import plugins
    
    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "build/src/tt/pjrt_plugin_tt.so"
            )
    
    plugins.register_plugin("TT", TTPjrtPlugin())
    xr.use_spmd()
    torch_xla.sync(True, True)


def create_mesh(mesh_shape: Tuple[int, int] = (1, 8)) -> Mesh:
    """
    Create device mesh for tensor parallelism.
    
    Args:
        mesh_shape: Tuple of (batch_dim, model_dim) for the device mesh
        
    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("batch", "model"))


class TensorParallelLinear(nn.Module):
    """
    A tensor parallel linear layer that shards weights across model dimension.
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
            partition_dim: Which dimension to partition (0 for row parallel, 1 for column parallel)
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.partition_dim = partition_dim
        self.mesh = mesh
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def apply_sharding(self, mesh: Mesh):
        """Apply tensor parallel sharding to weights."""
        if self.partition_dim == 0:
            # Row parallel - shard output dimension
            xs.mark_sharding(self.weight, mesh, ("model", None))
        else:
            # Column parallel - shard input dimension  
            xs.mark_sharding(self.weight, mesh, (None, "model"))
            
        if self.bias is not None:
            if self.partition_dim == 0:
                xs.mark_sharding(self.bias, mesh, ("model",))
            else:
                # Column parallel bias is replicated
                xs.mark_sharding(self.bias, mesh, (None,))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, self.weight, self.bias)


class TensorParallelLlamaMLP(nn.Module):
    """
    Tensor parallel version of LlamaMLP with proper weight sharding.
    """
    
    def __init__(self, config: LlamaConfig, mesh: Optional[Mesh] = None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.mesh = mesh
        
        # Column parallel projections (shard output)
        self.gate_proj = TensorParallelLinear(
            self.hidden_size, self.intermediate_size, bias=False, 
            mesh=mesh, partition_dim=0
        )
        self.up_proj = TensorParallelLinear(
            self.hidden_size, self.intermediate_size, bias=False,
            mesh=mesh, partition_dim=0  
        )
        
        # Row parallel projection (shard input)
        self.down_proj = TensorParallelLinear(
            self.intermediate_size, self.hidden_size, bias=False,
            mesh=mesh, partition_dim=1
        )
        
        self.act_fn = nn.SiLU()
    
    def apply_sharding(self, mesh: Mesh):
        """Apply tensor parallel sharding."""
        self.gate_proj.apply_sharding(mesh)
        self.up_proj.apply_sharding(mesh)
        self.down_proj.apply_sharding(mesh)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Column parallel forward
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x) 
        
        # Apply activation
        intermediate = self.act_fn(gate_out) * up_out
        
        # Row parallel forward with allreduce
        output = self.down_proj(intermediate)
        
        return output


class TensorParallelLlamaAttention(nn.Module):
    """
    Tensor parallel version of LlamaAttention with proper weight sharding.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None, 
                 mesh: Optional[Mesh] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mesh = mesh
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        # Column parallel projections
        self.q_proj = TensorParallelLinear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False,
            mesh=mesh, partition_dim=0
        )
        self.k_proj = TensorParallelLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False,
            mesh=mesh, partition_dim=0
        )
        self.v_proj = TensorParallelLinear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False,
            mesh=mesh, partition_dim=0
        )
        
        # Row parallel projection
        self.o_proj = TensorParallelLinear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False,
            mesh=mesh, partition_dim=1
        )
    
    def apply_sharding(self, mesh: Mesh):
        """Apply tensor parallel sharding."""
        self.q_proj.apply_sharding(mesh)
        self.k_proj.apply_sharding(mesh) 
        self.v_proj.apply_sharding(mesh)
        self.o_proj.apply_sharding(mesh)
    
    def _repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).
        The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
        (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        bsz, q_len, _ = hidden_states.size()
        
        # Column parallel projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Get local head counts after sharding
        local_num_heads = query_states.shape[-1] // self.head_dim
        local_num_kv_heads = key_states.shape[-1] // self.head_dim
        
        # Reshape for attention
        query_states = query_states.view(bsz, q_len, local_num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, local_num_kv_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, local_num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embedding if provided
        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Repeat k/v for local attention heads
        key_states = self._repeat_kv(key_states, self.num_key_value_groups)
        value_states = self._repeat_kv(value_states, self.num_key_value_groups)
        
        # Attention computation
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        
        # Row parallel projection with allreduce
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Apply rotary positional embedding to query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class TensorParallelLlamaDecoderLayer(nn.Module):
    """
    Tensor parallel version of LlamaDecoderLayer.
    """
    
    def __init__(self, config: LlamaConfig, layer_idx: int, mesh: Optional[Mesh] = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.mesh = mesh
        
        self.self_attn = TensorParallelLlamaAttention(config, layer_idx, mesh)
        self.mlp = TensorParallelLlamaMLP(config, mesh)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def apply_sharding(self, mesh: Mesh):
        """Apply tensor parallel sharding."""
        self.self_attn.apply_sharding(mesh)
        self.mlp.apply_sharding(mesh)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor]:
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return (hidden_states,)


def convert_llama_to_tensor_parallel(model: LlamaModel, mesh: Mesh) -> LlamaModel:
    """
    Convert a standard Llama model to use tensor parallelism.
    
    Args:
        model: Standard Llama model
        mesh: Device mesh for tensor parallelism
        
    Returns:
        Model with tensor parallel layers
    """
    
    # Convert each decoder layer
    for i, layer in enumerate(model.layers):
        # Create tensor parallel layer
        tp_layer = TensorParallelLlamaDecoderLayer(model.config, i, mesh)
        
        # Copy weights from original layer
        tp_layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data
        tp_layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data  
        tp_layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data
        tp_layer.self_attn.o_proj.weight.data = layer.self_attn.o_proj.weight.data
        
        tp_layer.mlp.gate_proj.weight.data = layer.mlp.gate_proj.weight.data
        tp_layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data
        tp_layer.mlp.down_proj.weight.data = layer.mlp.down_proj.weight.data
        
        tp_layer.input_layernorm.weight.data = layer.input_layernorm.weight.data
        tp_layer.post_attention_layernorm.weight.data = layer.post_attention_layernorm.weight.data
        
        # Apply sharding
        tp_layer.apply_sharding(mesh)
        
        # Replace layer in model
        model.layers[i] = tp_layer
    
    return model


def shard_llama_weights(model: LlamaModel, mesh: Mesh):
    """
    Apply tensor parallel sharding to an existing Llama model.
    
    Args:
        model: Llama model to shard
        mesh: Device mesh for sharding
    """
    
    # Move model to XLA device first
    model = model.to(torch_xla.device())
    
    # Apply sharding to each layer
    for layer in model.layers:
        # MLP weight sharding
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))  
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))
        
        # Attention weight sharding
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))


# Example usage function
def main():
    """Example of how to use tensor parallel Llama."""
    
    # Setup environment
    setup_tensor_parallel_environment()
    mesh = create_mesh()
    
    # Load model
    config = LlamaConfig.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    model = LlamaModel(config)
    
    # Apply tensor parallelism
    shard_llama_weights(model, mesh)
    
    # Example forward pass
    batch_size = 1
    seq_len = 1024
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    input_ids = input_ids.to(torch_xla.device())
    
    # Mark input sharding (replicated across model dimension)
    xs.mark_sharding(input_ids, mesh, (None, None))
    
    # Forward pass
    outputs = model(input_ids=input_ids)
    
    print(f"Output shape: {outputs.last_hidden_state.shape}")
    print("Tensor parallel forward pass completed!")


if __name__ == "__main__":
    main()
