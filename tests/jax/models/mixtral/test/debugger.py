import torch
from transformers.activations import ACT2FN
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os
from flax import nnx
from singlechip.flaxconfigmixtral import MixtralConfig
from jax.experimental.shard_map import shard_map
from jax_config import cpu_devices, axis_name, num_devices, device_mesh
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax import lax

def pcc(x, y):
    if hasattr(x, 'numpy'):
        x = x.numpy()  # PyTorch tensor
    if hasattr(y, 'numpy'):
        y = y.numpy()  # PyTorch tensor
    
    # Convert JAX arrays to numpy if needed
    if isinstance(x, jnp.ndarray):
        x = np.array(x)
    if isinstance(y, jnp.ndarray):
        y = np.array(y)
    
    # Flatten tensors
    x_flat = x.flatten()
    y_flat = y.flatten()
    
    # Calculate correlation
    correlation = np.corrcoef(x_flat, y_flat)[0, 1]
    
    return correlation



config = MixtralConfig(num_hidden_layers=1, intermediate_size=4096)
prng_key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(0)
batch_size = 8
seq_len = 10
tokens = 5
hidden_size = 4096
max_len = seq_len + tokens
input_data = jax.random.normal(key = prng_key, shape = (batch_size, seq_len, hidden_size))
attention_mask = jnp.ones((batch_size, seq_len), dtype = jnp.int32)
model = MixtralSparseMoeBlock(config, dtype = jnp.float32, rngs = rngs)

def runner(input_data, attention_mask, max_len):
    print("Creating sharded model...")
    print("Sharded Model created")
    batch_size, seq_len, hidden_size = input_data.shape
    
    # Handle padding - this is fine during compilation since it's based on shape
    inputs_spec = P("X")  
    pad_size = 0
    if batch_size % num_devices != 0:
        pad_size = num_devices - (batch_size % num_devices)
        padding = jnp.zeros((pad_size, seq_len), dtype=input_data.dtype)
        input_data = jnp.concatenate([input_data, padding], axis=0)
        
        mask_padding = jnp.zeros((pad_size, seq_len), dtype=attention_mask.dtype)
        attention_mask = jnp.concatenate([attention_mask, mask_padding], axis=0)
        
        batch_size += pad_size

    # Same sharding setup
    sharded_input = jax.device_put(input_data, NamedSharding(device_mesh, inputs_spec))
    out_spec = P()
    
    # Set up attention mask
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype="i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    sharded_mask = jax.device_put(extended_attention_mask, NamedSharding(device_mesh, P("X", None)))

    # KV Cache setup - same as before
    past_key_values = {}
    for i in range(config.num_hidden_layers):
        layer_key = f'layer_{i}'
        past_key_values[layer_key] = {
            'cached_key': jnp.zeros((batch_size, max_len, 8, 128), dtype=jnp.float32),
            'cached_value': jnp.zeros((batch_size, max_len, 8, 128), dtype=jnp.float32),
            'cache_index': jnp.array(0, dtype=jnp.int32)
        }
        
    sharded_cache = {}
    cache_specs = {}
    
    for layer_key, layer_cache in past_key_values.items():
        sharded_cache[layer_key] = {}
        sharded_cache[layer_key]['cached_key'] = jax.device_put(
            layer_cache['cached_key'],
            NamedSharding(device_mesh, P("X", None, None, None))
        )
        sharded_cache[layer_key]['cached_value'] = jax.device_put(
            layer_cache['cached_value'],
            NamedSharding(device_mesh, P("X", None, None, None))
        )
        sharded_cache[layer_key]['cache_index'] = jax.device_put(
            layer_cache['cache_index'],
            NamedSharding(device_mesh, P())
        )
        
    for i in range(config.num_hidden_layers):
        layer_key = f'layer_{i}'
        cache_specs[layer_key] = {
            'cached_key': P("X", None, None, None),
            'cached_value': P("X", None, None, None),
            'cache_index': P()
        }
        
    position_ids = jnp.cumsum(extended_attention_mask, axis=-1) - 1
    sharded_position_ids = jax.device_put(position_ids, NamedSharding(device_mesh, P("X", None)))

    def forward_pass(x):
        # Single forward pass through the model
        outputs = model(
            hidden_states=x,
            # attention_mask=mask,
            # past_key_values=cache,
            # position_ids=pos,
            # init_cache=True,
        )
        return outputs
    
    # Create sharded and JIT-compiled forward function
    sharded_forward = shard_map(
        forward_pass,
        device_mesh,
        in_specs=(inputs_spec), 
        out_specs=(P("X", None, None)),  # logits and updated cache
        check_rep=False,
    )
    
    # JIT compile the sharded forward pass
    jit_forward = jax.jit(sharded_forward)
    
    # Manual generation loop (outside JIT)
    # all_token_ids = jnp.zeros((batch_size, max_len), dtype=input_data.dtype)
    # all_token_ids = all_token_ids.at[:, :seq_len].set(sharded_input)
    
    # Process initial prompt
    logits = jit_forward(
        sharded_input, 
        # sharded_mask, 
        # sharded_cache, 
        # sharded_position_ids
    )
    return logits

res = runner(input_data, attention_mask, max_len)
print(res.shape)
print(res)

class MixtralBlockSparseTop2MLP2(nnx.Module):
    """MLP module with sparse routing for Mixtral architecture."""
    
    def __init__(self, config: MixtralConfig, rngs : nnx.Rngs):
        super().__init__()
        embed_dim = config.hidden_size
        inner_dim = config.intermediate_size if config.intermediate_size is not None else 4 * embed_dim
        self.up_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False, rngs = rngs)
        self.gate_proj = nnx.Linear(embed_dim, inner_dim, use_bias=False, rngs = rngs)
        self.down_proj = nnx.Linear(inner_dim, embed_dim, use_bias=False, rngs = rngs)
        self.act_fn = jax.nn.silu

    def __call__(self, hidden_states):
        gate_states = self.act_fn(self.up_proj(hidden_states)) * self.gate_proj(hidden_states)
        hidden_states = self.down_proj(gate_states)
        return hidden_states

class MixtralSparseMoeBlock2(nnx.Module):
    """Sparse Mixture of Experts block for Mixtral."""
    
    def __init__(self, config: MixtralConfig, dtype, rngs : nnx.Rngs):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        self.dtype = dtype
        self.gate = nnx.Linear(
            config.hidden_size, 
            config.num_local_experts, 
            use_bias=False, 
            dtype=self.dtype, 
            rngs = rngs
        )

        for i in range(self.num_experts):
            setattr(self, f"experts_{i}", MixtralBlockSparseTop2MLP(config, rngs = rngs))
        self.jitter_noise = config.router_jitter_noise

    def _get_expert(self, idx):
        """Helper to get expert by index"""
        return getattr(self, f"experts_{idx}")

    def __call__(self, hidden_states):
        batch_size, seq_len, hid_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hid_dim)
        router_logits = self.gate(hidden_states)
        routing_weights = jax.nn.softmax(router_logits, axis = 1)
        routing_weights, selected_experts = lax.top_k(routing_weights, self.top_k)
        routing_weights /= jnp.sum(routing_weights, axis = -1, keepdims = True)
        # print(routing_weights, selected_experts)
        routing_weights = routing_weights.astype(hidden_states.dtype)

        final_hidden_states = jnp.zeros(
            (batch_size * seq_len, hid_dim), dtype=hidden_states.dtype)

        # Create one-hot representation of selected experts
        expert_mask = jax.nn.one_hot(selected_experts, num_classes = self.num_experts, dtype = jnp.int8).transpose(2, 1, 0)
        for expert_idx in range(self.num_experts):
            expert_layer = self._get_expert(expert_idx)
            idx, top_x = jnp.where(expert_mask[expert_idx])
            current_state = hidden_states[None, top_x].reshape(-1, hid_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states = final_hidden_states.at[top_x].add(current_hidden_states)

        final_hidden_states = final_hidden_states.reshape(batch_size, seq_len, hid_dim)
        # print(final_hidden_states)
        return final_hidden_states, router_logits

model2 = MixtralSparseMoeBlock2(config, dtype = jnp.float32, rngs = rngs)
def run_single_chip(input_data, attention_mask, max_len):
    print("Creating Regular model...")
    model2.gate.kernel.value = model.gate.kernel.value
    for i in range(8):
        expert2 = getattr(model2, f"experts_{i}")
        expert = getattr(model, f"experts_{i}")
        expert2.up_proj.kernel.value = expert.up_proj.kernel.value
        expert2.down_proj.kernel.value = expert.down_proj.kernel.value
        expert2.gate_proj.kernel.value = expert.gate_proj.kernel.value
    print("Regular model created")
    batch_size, seq_len, hidden_size = input_data.shape
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype = "i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))

    out1, out2 = model2(
        hidden_states = input_data
    )
    return out1

res2 = run_single_chip(input_data, attention_mask, max_len)
print(res2.shape)
print(res2)

print(pcc(res, res2))
print(pcc(model2.gate.kernel.value, model.gate.kernel.value))