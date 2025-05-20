import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.activations import ACT2FN
from huggingface_hub import login
import numpy as np
import jax
import jax.numpy as jnp
import os
from flax import nnx
import torch.nn.functional as F
from torch import nn
from typing import Callable, List, Optional, Tuple, Union
from singlechip.convert_weights import make_model

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

np_dir = "mixtral_numpy_weights"
os.makedirs(np_dir, exist_ok=True)
#removed the token
# with open('tokens.txt') as f:
#     hf_token = f.read()

# Log in to Hugging Face Hub with your token
login(token=hf_token)  # Replace with your actual token


# Print status
print("Loading model - this may take several minutes...")
model_id = "mistralai/Mixtral-8x7B-v0.1"
config = AutoConfig.from_pretrained(model_id)
config.num_hidden_layers = 1
# Load the model with quantization

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    device_map="auto",
    torch_dtype=torch.float32
)
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# config_dict = config.to_dict()
# with open(f"{np_dir}/config.json", "w") as f:
#     import json
#     json.dump(config_dict, f, indent=2)
# print(f"Saved configuration to {np_dir}/config.json")

# # 4. First, let's examine the model structure to verify the component names
# layer0 = model.model.layers[0]

# # Print the keys of the first layer's state dict to understand the structure
# # layer0_keys = layer0.state_dict().keys()

# # Save embeddings
# embeddings = model.model.embed_tokens.weight.detach().cpu().numpy()
# np.save(f"{np_dir}/embeddings.npy", embeddings)

# # Save LM head
# lm_head = model.lm_head.weight.detach().cpu().numpy()
# np.save(f"{np_dir}/lm_head.npy", lm_head)

# # Save attention components
# attn_q = layer0.self_attn.q_proj.weight.detach().cpu().numpy()
# attn_k = layer0.self_attn.k_proj.weight.detach().cpu().numpy()
# attn_v = layer0.self_attn.v_proj.weight.detach().cpu().numpy()
# attn_o = layer0.self_attn.o_proj.weight.detach().cpu().numpy()

# np.save(f"{np_dir}/attn_q_proj.npy", attn_q)
# np.save(f"{np_dir}/attn_k_proj.npy", attn_k)
# np.save(f"{np_dir}/attn_v_proj.npy", attn_v)
# np.save(f"{np_dir}/attn_o_proj.npy", attn_o)

# # Save MoE experts correctly - inspect the structure first
# moe = layer0.block_sparse_moe
# num_experts = config.num_local_experts

# # Get all experts' weights
# for i in range(num_experts):
#     # Correct structure for Mixtral's experts
#     w1 = moe.experts[i].w1.weight.detach().cpu().numpy()
#     w2 = moe.experts[i].w2.weight.detach().cpu().numpy()
#     w3 = moe.experts[i].w3.weight.detach().cpu().numpy()
    
#     np.save(f"{np_dir}/expert_{i}_w1.npy", w1)
#     np.save(f"{np_dir}/expert_{i}_w2.npy", w2)
#     np.save(f"{np_dir}/expert_{i}_w3.npy", w3)

# # Save router/gate weights
# gate = moe.gate.weight.detach().cpu().numpy()
# np.save(f"{np_dir}/moe_gate.npy", gate)

# # Save layer norms
# input_layernorm = layer0.input_layernorm.weight.detach().cpu().numpy()
# post_attention_layernorm = layer0.post_attention_layernorm.weight.detach().cpu().numpy()
# final_norm = model.model.norm.weight.detach().cpu().numpy()

# np.save(f"{np_dir}/input_layernorm.npy", input_layernorm)
# np.save(f"{np_dir}/post_attention_layernorm.npy", post_attention_layernorm)
# np.save(f"{np_dir}/final_norm.npy", final_norm)

# # Create a metadata file
# metadata = {
#     "model": model_id,
#     "version": "single-layer",
#     "vocab_size": config.vocab_size,
#     "hidden_size": config.hidden_size,
#     "intermediate_size": config.intermediate_size,
#     "num_attention_heads": config.num_attention_heads,
#     "num_key_value_heads": config.num_key_value_heads,
#     "num_experts": config.num_local_experts,
#     "num_experts_per_tok": config.num_experts_per_tok,
#     "files": [f for f in os.listdir(np_dir) if f.endswith('.npy')],
#     "parameter_count": sum(p.numel() for p in model.parameters()),
#     "component_shapes": {
#         "embeddings": embeddings.shape,
#         "lm_head": lm_head.shape,
#         "attn_q_proj": attn_q.shape,
#         "attn_k_proj": attn_k.shape,
#         "attn_v_proj": attn_v.shape,
#         "attn_o_proj": attn_o.shape,
#         "expert_w1": w1.shape,  # Same for all experts
#         "expert_w2": w2.shape,
#         "expert_w3": w3.shape,
#         "moe_gate": gate.shape,
#         "input_layernorm": input_layernorm.shape,
#         "post_attention_layernorm": post_attention_layernorm.shape,
#         "final_norm": final_norm.shape,
#     }
# }

# with open(f"{np_dir}/metadata.json", "w") as f:
#     import json
#     json.dump(metadata, f, indent=2)

# print(f"\nSuccessfully saved all weights as NumPy files to {np_dir}/")
# print(f"Total parameter count: {metadata['parameter_count']:,}")

# # Example prompt
prompt = "The capital city of USA is "
inputs = tokenizer(prompt, return_tensors="pt")
print("Model created successfully!")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
np.save("hf_input_ids.npy", input_ids.numpy())
np.save("hf_attention_mask.npy", attention_mask.numpy())

# # Function to convert PyTorch tensor to JAX array
# for name, param in model.named_parameters():
#     # Convert to numpy array
#     param_numpy = param.detach().cpu().numpy()
    
#     # Create directory structure if needed
#     parts = name.split('.')
#     directory = os.path.join("mixtral_numpy_params", *parts[:-1])
#     os.makedirs(directory, exist_ok=True)
    
#     # Save as .npy file (efficient binary NumPy format)
#     filename = os.path.join("mixtral_numpy_params", *parts) + ".npy"
#     np.save(filename, param_numpy)
    
    # print(f"Saved {name} with shape {param_numpy.shape} to {filename}")

# with open("mixtral_numpy_params/manifest.txt", "w") as f:
#     for name, param in model.named_parameters():
#         f.write(f"{name},{','.join(str(dim) for dim in param.shape)}\n")

# print("All parameters saved!")
print("Generating respose...")

def compare_attention_with_manual_input():
    """Compare attention using manually created inputs"""
    print("Testing attention with manual inputs...")
    
    # --- Create manual inputs ---
    # Set dimensions based on Mixtral's configuration
    batch_size = 2
    seq_len = 10
    hidden_size = 4096  # Mixtral's hidden size
    
    # Create a simple random input tensor that will be identical for both models
    np.random.seed(42)  # Set seed for reproducibility
    manual_input_np = np.random.randn(batch_size, seq_len, hidden_size)
    # Create attention mask (all 1's for simplicity)
    attention_mask_np = np.ones((batch_size, seq_len))
    
    print(f"Created manual input with shape: {manual_input_np.shape}")
    
    # --- Load Flax model ---
    print("Loading Flax model...")
    
    flax_model = make_model(config, model)  # Your loading function
    
    # --- Convert inputs to appropriate formats ---
    # For PyTorch
    manual_input_torch = torch.tensor(manual_input_np, dtype=torch.long).to(model.device)
    attention_mask_torch = torch.tensor(attention_mask_np).to(model.device)
    
    # For JAX
    manual_input_jax = jnp.array(manual_input_np).astype(jnp.int32)
    attention_mask_jax = jnp.array(attention_mask_np)

    # --- Compare Q, K, V projections ---
    print("\nComparing Q, K, V projections...")
    
    hf_layer = model.model.layers[0]
    flax_layer = flax_model.model.layers[0]

        # print(f"Q projection correlation: {q_corr}")
        # print(f"K projection correlation: {k_corr}")
        # print(f"V projection correlation: {v_corr}")
    
        # print("\nProjection shapes:")
        # print(f"HF Q: {hf_w1.shape}, Flax Q: {flax_w1.shape}")
        # print(f"HF K: {hf_w2.shape}, Flax K: {flax_w2.shape}")
        # print(f"HF V: {hf_w3.shape}, Flax V: {flax_w3.shape}")
    
    
    try:
        print(manual_input_torch.shape, manual_input_jax.shape)

        with torch.no_grad():
            hf_attn_output = hf_layer(
                hidden_states = manual_input_torch,
                # position_ids = position_ids_torch
                attention_mask=attention_mask_torch,
                #position_embeddings = hf_position_embeddings,
                # output_attentions=True
            )
        print(hf_attn_output.shape)
    #     # Flax attention output
    #     # attention_mask_jax = attention_mask_jax.reshape(1, 1, batch_size, seq_len)

        flax_attn_output =flax_layer(
            hidden_states = manual_input_jax,
            attention_mask = attention_mask_jax
        )
        print(hf_attn_output)
        print(flax_attn_output.shape)
        # sga.experts[0].w1.weight.data = hf_w1
        print(flax_attn_output)
        hf_attn_output = hf_attn_output.detach()
        print(pcc(flax_attn_output, hf_attn_output))
            # expert.gate_proj.kernel.value = match_shape(
            #     expert.gate_proj.kernel.value, 
            #     np_weights["experts"][e]["w3"]
            # )
            # expert.up_proj.kernel.value = match_shape(
            #     expert.up_proj.kernel.value, 
            #     np_weights["experts"][e]["w1"]
            # )
            # expert.down_proj.kernel.value = match_shape(
            #     expert.down_proj.kernel.value, 
            #     np_weights["experts"][e]["w2"]
            # )
    #     attn = MixtralAttention(config, 0)
    #     
    #     attn.o_proj.weight.data = torch.tensor(attn_o)
    #     with torch.no_grad():
    #         outattn = attn(hidden_states=manual_input_torch,
    #                 attention_mask=attention_mask,
    #                 position_embeddings = hf_position_embeddings,
    #                 gas = flax_attn_output,
    #         )

    #     print(outattn[0].shape)
    #     print(hf_attn_output.shape)
    #     print(flax_attn_output.shape)
    #     # Compare with PCC
    #     attn_corr = pcc(hf_attn_output, outattn[0])
    #     print(f"Full attention output correlation: {attn_corr}")
    #     print("AATn2: ", pcc(flax_attn_output, outattn[0]))
    #     # Calculate differences
    
        
    #     # Sample values
    #     print("\nSample attention output values (first token, first 5 values):")
    #     print(f"HF:   {hf_attn_output[0, 0, :5]}")
    #     print(f"Flax: {flax_attn_output[0, 0, :5]}")
        
    except Exception as e:
        print(f"Error running attention comparison: {e}")
        import traceback
        traceback.print_exc()
        
        # Try running with different signatures or without attention mask
        print("\nTrying alternative attention call patterns...")
    
    # return {
    #     "q_corr": q_corr,
    #     "k_corr": k_corr,
    #     "v_corr": v_corr,
    #     "hf_q": hf_q,
    #     "flax_q": flax_q,
    #     "hf_k": hf_k,
    #     "flax_k": flax_k,
    #     "hf_v": hf_v,
    #     "flax_v": flax_v
    # }
results = compare_attention_with_manual_input()

# print('Token ids:')
# print(output[0])
# # # Decode the output
# generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# print("\nRESPONSE:")
# print(generated_text)
# np.save("output.npy", output[0].numpy())