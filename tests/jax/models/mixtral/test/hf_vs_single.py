import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys
from dotenv import load_dotenv
from jax import lax

load_dotenv()

from flax import nnx
from torch import nn
from singlechip.convert_weights import make_model

#need a token
hf_token = os.getenv("HF_TOKEN")
login(token=hf_token) 


def main(model_id = 'mistralai/Mixtral-8x7B-v0.1',
    prompt: str = (
    "Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. "
    "She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
    "How much in dollars does she make every day at the farmers' market?\n"
    "A: ")):

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    config.num_hidden_layers = 2
    config._attn_implementation = "eager"
    # Print all special tokens and their IDs
    for token_name in tokenizer.special_tokens_map:
        token_str = tokenizer.special_tokens_map[token_name]
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        print(f"{token_name}: {token_str} -> {token_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32
    )
    print(tokenizer.eos_token, tokenizer.pad_token)
    print(tokenizer.special_tokens_map)

    inputs = tokenizer(prompt, return_tensors="pt")
    batch_size = 3
    seq_len = 10
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)
    print("eos_token_id in input_ids:", tokenizer.eos_token_id in input_ids[0])
    input_ids = torch.randint(config.vocab_size, (8, 10)).long()
    attention_mask = torch.ones_like(input_ids)
    print("✍️ Generating...")
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask
    )
    max_len = outputs[0].shape[0]
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>"):].lstrip()

    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()

    
    batch_size, seq_len = input_ids.shape
    input_ids = jnp.array(input_ids)
    attention_mask = jnp.array(attention_mask)
    flax_model = make_model(config, model)
    
    for i in range(config.num_hidden_layers):
        flax_model.model.layers[i].attn.cached_key = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        flax_model.model.layers[i].attn.cached_value = jnp.zeros((batch_size, max_len, 8, 128), dtype = jnp.float32)
        flax_model.model.layers[i].attn.cache_index = jnp.array(0, dtype = jnp.int32)

    extended_attention_mask = jnp.ones((batch_size, max_len), dtype = "i4")
    extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask, (0, 0))
    # extended_attention_mask = lax.dynamic_update_slice(extended_attention_mask, attention_mask_jax, (0, 0))
    generated_ids = flax_model.generate(
        input_ids=input_ids,
        attention_mask=extended_attention_mask,
        max_new_tokens=max_len - seq_len 
    )
    print(generated_ids.shape)
    print(generated_ids)
    resultJax = tokenizer.decode(torch.tensor(generated_ids[0]), skip_special_tokens=False)
    if resultJax.startswith("<|begin_of_text|>"):
        resultJax = resultJax[len("<|begin_of_text|>"):].lstrip()

    if result.startswith(prompt):
        resultJax = resultJax[len(prompt):].lstrip()
    print("OutputJax:", resultJax)

    print("\n🧠 OutputHF:\n", result) 
    print(np.array(resultJax) == np.array(result))
    
main()
