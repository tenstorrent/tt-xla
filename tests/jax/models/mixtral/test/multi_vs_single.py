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
from transformers import AutoConfig

from jax_config import num_devices, device_mesh




def run_single_chip(input_ids, attention_mask, max_len, config):
    print("Creating Regular model...")
    model = NotShardedModel(config)
    print("Regular model created")
    input_ids, attention_mask, position_ids = model.prepare_inputs_for_generation(input_ids, max_len, attention_mask)
    out = model.generate(
        input_ids = input_ids,
        attention_mask = attention_mask,
        max_new_tokens = max_len - seq_len,
        position_ids = position_ids 
    )
    return out

def create_sharded_model(model):
    partitioning_rules = {
        # Attention layers - tensor parallel
        ('model', 'layers', '*', 'attn', 'q_proj', 'kernel'): P(None, 'X'),
        ('model', 'layers', '*', 'attn', 'k_proj', 'kernel'): P(None, 'X'),
        ('model', 'layers', '*', 'attn', 'v_proj', 'kernel'): P(None, 'X'),
        ('model', 'layers', '*', 'attn', 'o_proj', 'kernel'): P('X', None),
        
        # MoE layers - tensor parallel
        ('model', 'layers', '*', 'block_sparse_moe', 'gate', 'kernel'): P(),  # Router replicated
        ('model', 'layers', '*', 'block_sparse_moe', 'experts', '*', 'up_proj', 'kernel'): P(None, 'X'),
        ('model', 'layers', '*', 'block_sparse_moe', 'experts', '*', 'down_proj', 'kernel'): P('X', None),
        ('model', 'layers', '*', 'block_sparse_moe', 'experts', '*', 'gate', 'kernel'): P(None, 'X'),
        
        # Embeddings and output layers
        ('model', 'embed_tokens', 'embedding'): P('X', None),
        ('lm_head', 'kernel'): P(None, 'X'),
        
        # Layer norms stay replicated
        ('model', 'layers', '*', 'input_norm'): P(),
        ('model', 'layers', '*', 'attn_norm'): P(),
        ('model', 'norm'): P(),
    }
    
    sharded_model = nnx.with_partitioning(model, partitioning_rules)
    
    return sharded_model

def run_multi_chip(input_ids, attention_mask, max_len, config):
    print("Creating sharded model...")
    model = ShardedModel(config)
    print("Sharded Model created")
    batch_size, seq_len = input_ids.shape
    input_ids, attention_mask, position_ids, past_key_values = model.prepare_inputs_for_generation(input_ids, max_len, attention_mask)    

    sharded_model = create_sharded_model(model)

    cache_specs = {}
    for i in range(config.num_hidden_layers):
        layer_key = f'layer_{i}'
        cache_specs[layer_key] = {
            'cached_key': P(),     # Replicated
            'cached_value': P(),   # Replicated
            'cache_index': P()     # Replicated
        }

    def model_forward(input_ids, attention_mask, position_ids, past_key_values):
        # Call the model
        outputs = sharded_model(
            input_ids,
            attention_mask,
            position_ids,
            past_key_values,
        )
        outputs = outputs.raw_value
        return outputs.logits, outputs.past_key_values

    sharded_forward = nnx.shard_map(
        model_forward,
        mesh = device_mesh,
        in_specs = (P(), P(), P(), cache_specs), 
        out_specs = (P(), cache_specs),          
        check_rep = False
    )

    jit_forward = nnx.jit(sharded_forward, donate_argnums=(3,))
    
    all_token_ids = jnp.zeros((batch_size, max_len), dtype=input_ids.dtype)
    all_token_ids = all_token_ids.at[:, :seq_len].set(input_ids)
    
    # First forward pass
    logits, past_key_values = jit_forward(
        input_ids, 
        attention_mask, 
        position_ids[:, :seq_len],
        past_key_values 
    )
    next_token_logits = logits[:, -1, :]
    next_token = jnp.argmax(next_token_logits, axis=-1)
    all_token_ids = all_token_ids.at[:, seq_len].set(next_token)
    return all_token_ids
    # def generate_step(i, carry):
    #     all_tokens, cache = carry
    #     cur_len = seq_len + 1 + i
        
    #     next_token_input = lax.dynamic_slice(
    #         all_tokens, 
    #         (0, cur_len-1), 
    #         (batch_size, 1)
    #     )
        
    #     new_position_ids = lax.dynamic_slice(
    #         position_ids,
    #         (0, cur_len-1),
    #         (batch_size, 1)
    #     )
        
    #     # Forward pass
    #     logits, updated_cache = jit_forward(
    #         next_token_input,
    #         attention_mask,
    #         cache,
    #         new_position_ids
    #     )
        
    #     next_token_logits = logits[:, -1, :]
    #     next_token = jnp.argmax(next_token_logits, axis=-1)
        
    #     updated_tokens = lax.dynamic_update_slice(
    #         all_tokens,
    #         next_token[:, None],  
    #         (0, cur_len)
    #     )
        
    #     return updated_tokens, updated_cache
    
    # num_new_tokens = max_len - seq_len - 1
    # final_tokens, _ = lax.fori_loop(
    #     0, 
    #     num_new_tokens,
    #     generate_step,
    #     (all_token_ids, cache_specs)
    # )
    
    # return final_tokens
  
if __name__ == '__main__':
    model_id = "mistralai/Mixtral-8x7B-v0.1"
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2
    config._attn_implementation = "eager"
    config.intermediate_size = 1024
    print(config)
    prng_key = jax.random.PRNGKey(0)
    rngs = nnx.Rngs(0)

    batch_size = 1
    seq_len = 10
    tokens = 1

    max_len = seq_len + tokens
    input_data = jax.random.randint(key = prng_key, shape = (batch_size, seq_len), minval = 0, maxval = config.vocab_size)
    attention_mask = jnp.ones_like(input_data)
    single_chip = run_single_chip(input_data, attention_mask, max_len, config)
    multi_chip = run_multi_chip(input_data, attention_mask, max_len, config)
    print(single_chip)
    print(multi_chip)
    if np.array_equal(np.array(single_chip), np.array(multi_chip)):
        print('Test successful!')
    else:
        print('Something doesnt work :(')