import jax
import jax.numpy as jnp
from flax.core import unfreeze, freeze
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from model.jax_config import create_device_mesh
from model.model_falcon3 import FlaxFalcon3ForCausalLM

def init_torch_model(model_name: str, config):
    """
    Initialize the PyTorch model with the given configuration.
    """
    torch_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        device_map="auto",
        torch_dtype=torch.float32,
    )
    return torch_model

def init_flax_model(config, batch_size, max_len, checkpoint_path = None):
    """
    Initialize the Flax model from the PyTorch model.
    """
    flax_model = FlaxFalcon3ForCausalLM(config)
    flax_params = flax_model.convert_from_hf_weights(
        config=config,
        checkpoint_path=checkpoint_path,
        batch_size=batch_size,
        max_len=max_len
    )
    return flax_model, flax_params

def prepare_torch_input(model_name, prompt):
    """
    Prepare input for the PyTorch model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    
    return tokenizer, inputs.input_ids, inputs.attention_mask

def prepare_flax_input(flax_model, input_ids, attention_mask, max_len):
    """
    Prepare input for the Flax model.
    """
    input_ids = jnp.array(input_ids.numpy(), dtype=jnp.int32)
    input_ids = jnp.repeat(input_ids, 2, axis=0)  # Duplicate for batch size of 2
    attention_mask = jnp.array(attention_mask.numpy(), dtype=jnp.int32)
    attention_mask = jnp.repeat(attention_mask, 2, axis=0)
    inputs = flax_model.prepare_inputs_for_generation(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=max_len,
    )
    return input_ids, inputs['attention_mask'], inputs['position_ids']

def run_torch_model(torch_model, input_ids, attention_mask):
    """
    Run the PyTorch model with the given input IDs and attention mask.
    """
    print("🏢 Generating HF Model output...")
    outputs = torch_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,

    )
    return outputs

def run_flax_model(flax_params, flax_model, input_ids, attention_mask, position_ids, max_len):
    """
    Run the Flax model with the given input IDs and attention mask.
    """
    print("✍️ Generating Flax Model output...")
    _, seq_len = input_ids.shape
    jit_generate = jax.jit(
        flax_model.generate,
        static_argnames=('max_new_tokens', 'return_dict')
    )
    flax_params = unfreeze(flax_params)
    token_ids = jit_generate(
        params=flax_params,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_new_tokens=max_len-seq_len,
        return_dict=True,
    )
    return token_ids

def strip_output(result, prompt: str = "") -> str:
    """
    Strip the output of the model to remove any special tokens or leading text.
    """
    if result.startswith("<|begin_of_text|>"):
        result = result[len("<|begin_of_text|>"):].lstrip()
    if result.startswith(prompt):
        result = result[len(prompt):].lstrip()
    return result

def compare_results(torch_result: str, flax_result: str) -> str:
    print("HF Result:\n", torch_result)
    print("Flax Result:\n", flax_result)
    if torch_result == flax_result:
        return "✅ Outputs match!"
    else:
        return "❌ Outputs do not match!"


def run_test(model_name: str, prompt: str):
    """
    Run the test comparing Hugging Face and Flax models.
    """
    print("🪄  Initializing models...")
    config = AutoConfig.from_pretrained(
        model_name,
        num_hidden_layers=4,
        torch_dtype=torch.float32,
    )
    tokenizer, input_ids, attention_mask = prepare_torch_input(model_name, prompt)

    torch_model = init_torch_model(model_name, config)
    torch_output = run_torch_model(torch_model, input_ids, attention_mask)
    max_len = torch_output.shape[1]
    
    flax_model, flax_params = init_flax_model(
        config=config,
        batch_size=input_ids.shape[0] * 2,
        max_len=max_len,
        checkpoint_path=None  # Path to the checkpoint if needed
    )
    input_ids, attention_mask, position_ids = prepare_flax_input(
        flax_model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_len=max_len
    )
    print("🔄 Preparing inputs for Flax model...")
    print(f"Attention mask shape: {attention_mask.shape}")
    flax_output = run_flax_model(
        flax_params=flax_params,
        flax_model=flax_model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        max_len=max_len,
    )

    print("📦 Flax model output:", flax_output[0], sep='\n')
    print("📦 Torch model output:", torch_output[0], sep='\n')
    print("🈵 Decoding outputs...")
    torch_result = tokenizer.decode(torch_output[0], skip_special_tokens=False)
    #torch_result = strip_output(torch_result, prompt)
    flax_result = tokenizer.decode(flax_output[0], skip_special_tokens=False)
    #flax_result = strip_output(flax_result, prompt)

    print("🔍 Comparing outputs...")
    print(compare_results(torch_result, flax_result))


if __name__ == "__main__":
    run_test(
        model_name="tiiuae/Falcon3-7B-Instruct",
        prompt="""
Q: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four.
She sells the remainder at the farmers' market daily for $2 per fresh duck egg.
How much in dollars does she make every day at the farmers' market?\n
A: """
    )
