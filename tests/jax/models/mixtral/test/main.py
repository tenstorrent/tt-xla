import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os
import sys
from dotenv import load_dotenv

load_dotenv()

from flax import nnx
from torch import nn
from mymodel.convert_weights import make_model
hf_token = os.getenv("HF_TOKEN")

# Log in to Hugging Face Hub with your token
login(token=hf_token)  # Replace with your actual token

# Print status
print("Loading model - this may take several minutes...")
model_id = "mistralai/Mixtral-8x7B-v0.1"
config = AutoConfig.from_pretrained(model_id)
config.num_hidden_layers = 1
config._attn_implementation = "eager"
# Load the model with quantization

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto",
    torch_dtype=torch.float32
)
# Load tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_id)

# prompt = "Who are you? "
# inputs = tokenizer(prompt, return_tensors="pt")
input_ids = torch.randint(config.vocab_size, (1, 10)).long()
attention_mask = torch.ones_like(input_ids)
input_ids_torch = torch.tensor(input_ids).long()
attention_mask_torch = torch.tensor(attention_mask)
generated_ids_torch = model.generate(
    input_ids_torch,
    attention_mask=attention_mask_torch,
    max_new_tokens=5,  # Generate exactly 5 new tokens
)
print("Number of hidden_layers: ", config.num_hidden_layers)
print("Input:")
print(input_ids)
print("HF:")
print(generated_ids_torch)

# Run forward pass
flax_model = make_model(config, model)
rngs = nnx.Rngs(0)
input_ids_jax = jnp.array(input_ids)
attention_mask_jax = jnp.array(attention_mask)
# prepare = flax_model.prepare_inputs(
#     input_ids = input_ids_jax,
#     max_len = max_len,
#     attention_mask = attention_mask_jax
# )
# # Optional: Generate text           
prepare = flax_model.prepare_inputs(
      input_ids = input_ids_jax,
      max_len = 15,
      attention_mask = attention_mask_jax
)
print('Jax')
generated_ids = flax_model.generate(
    input_ids=input_ids_jax,
    attention_mask=prepare['attention_mask'],
    max_new_tokens=5
)
print("MyModel:")
print(generated_ids)
if np.array_equal(np.array(generated_ids_torch),np.array(generated_ids)):
    print('Test successful!')
else:
    print('Not working :(')