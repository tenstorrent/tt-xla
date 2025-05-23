import os
import jax
import jax.numpy as jnp
from flax import nnx
import flax.linen as nn
from jax.experimental.shard_map import shard_map
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax import lax
import numpy as np
from functools import partial

from singlechip.flaxmixtral import FlaxMixtralForCausalLM as NotShardedModel
from multichip.multichipmixtral import FlaxMixtralForCausalLM as ShardedModel
from singlechip.flaxconfigmixtral import MixtralConfig
from jax_config import cpu_devices, axis_name, num_devices, device_mesh

config = MixtralConfig(num_hidden_layers=2, intermediate_size=1024)
prng_key = jax.random.PRNGKey(0)
rngs = nnx.Rngs(0)


def run_single_chip(input_data, attention_mask, max_len):
    print("Creating Regular model...")
    model = NotShardedModel(config)
    print("Regular model created")
    batch_size, seq_len = input_data.shape
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype = "i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    for i in range(config.num_hidden_layers):
        model.model.layers[i].attn.cached_key = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        model.model.layers[i].attn.cached_value = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        model.model.layers[i].attn.cache_index = jnp.array(0, dtype = jnp.int32)

    out = model.generate(
        input_ids = input_data,
        attention_mask = extended_attention_mask,
        max_new_tokens = max_len - seq_len 
    )
    return out


def run_multi_chip(input_data, attention_mask, max_len):
    print("Creating sharded model...")
    model = ShardedModel(config)
    print("Sharded Model created")
    batch_size, seq_len = input_data.shape
    
    inputs_spec = P("X")  
    pad_size = 0
    if batch_size % num_devices != 0:
        pad_size = num_devices - (batch_size % num_devices)
        padding = jnp.zeros((pad_size, seq_len), dtype=input_data.dtype)
        input_data = jnp.concatenate([input_data, padding], axis=0)
        
        mask_padding = jnp.zeros((pad_size, seq_len), dtype=attention_mask.dtype)
        attention_mask = jnp.concatenate([attention_mask, mask_padding], axis=0)
        
        batch_size += pad_size

    sharded_input = jax.device_put(input_data, NamedSharding(device_mesh, inputs_spec))
    
    batch_size, seq_len = input_data.shape
    extended_attention_mask = jnp.ones((batch_size, max_len), dtype="i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    sharded_mask = jax.device_put(extended_attention_mask, NamedSharding(device_mesh, P("X", None)))

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
    position_ids_first = jnp.cumsum(attention_mask, axis = 1) - 1
    sharded_position_ids = jax.device_put(position_ids, NamedSharding(device_mesh, P("X", None)))
    sharded_position_ids_first = jax.device_put(position_ids_first, NamedSharding(device_mesh, P("X", None)))
    
    def forward_pass(x, mask, cache, pos):
        outputs = model(
            input_ids=x,
            attention_mask=mask,
            past_key_values=cache,
            position_ids=pos,
            init_cache=True,
        )
        return outputs.logits, outputs.past_key_values
    
    sharded_forward = shard_map(
        forward_pass,
        device_mesh,
        in_specs=(inputs_spec, P("X", None), cache_specs, P("X", None)), 
        out_specs=(P("X", None, None), cache_specs),  # logits and updated cache
        check_rep=False,
    )
    
    jit_forward = jax.jit(sharded_forward, donate_argnums=(2, ))
    
    all_token_ids = jnp.zeros((batch_size, max_len), dtype=input_data.dtype)
    all_token_ids = all_token_ids.at[:, :seq_len].set(sharded_input)
    
    logits, sharded_cache = jit_forward(
        sharded_input, 
        sharded_mask, 
        sharded_cache, 
        sharded_position_ids_first
    )
    
    next_token_logits = logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    all_token_ids = all_token_ids.at[:, seq_len].set(next_token)
    
    def generate_step(i, carry):
        all_tokens, cache = carry
        cur_len = seq_len + 1 + i
        
        next_token_input = lax.dynamic_slice(
            all_tokens, 
            (0, cur_len-1), 
            (batch_size, 1)
        )
        next_token_input = jax.device_put(
            next_token_input, 
            NamedSharding(device_mesh, P("X", None))
        )
        
        new_position_ids = lax.dynamic_slice(
            sharded_position_ids,
            (0, cur_len-1),
            (batch_size, 1)
        )
        
        # Forward pass
        logits, updated_cache = jit_forward(
            next_token_input,
            sharded_mask,
            cache,
            new_position_ids
        )
        
        next_token_logits = logits[:, -1, :]
        next_token = jnp.argmax(next_token_logits, axis=-1)
        
        updated_tokens = lax.dynamic_update_slice(
            all_tokens,
            next_token[:, None],  
            (0, cur_len)
        )
        
        return updated_tokens, updated_cache
    
    num_new_tokens = max_len - seq_len - 1
    final_tokens, _ = lax.fori_loop(
        0, 
        num_new_tokens,
        generate_step,
        (all_token_ids, sharded_cache)
    )
    
    # Remove padding if necessary
    if pad_size:
        original_batch_size = input_data.shape[0] - pad_size
        final_tokens = final_tokens[:original_batch_size]
    
    return final_tokens
  
if __name__ == '__main__':
    batch_size = 8
    seq_len = 10
    tokens = 100
    max_len = seq_len + tokens
    input_data = jax.random.randint(key = prng_key, shape = (batch_size, seq_len), minval = 0, maxval = config.vocab_size)
    attention_mask = jnp.ones_like(input_data)
    single_chip = run_single_chip(input_data, attention_mask, max_len)
    multi_chip = run_multi_chip(input_data, attention_mask, max_len)
    print(single_chip)
    print(multi_chip)
    if np.array_equal(np.array(single_chip), np.array(multi_chip)):
        print('Test successful!')
    else:
        print('Something doesnt work :(')