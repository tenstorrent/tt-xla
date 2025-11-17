# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from transformers import AutoTokenizer
import numpy as np
import flax.nnx as nnx


def create_model_on_device(model_name, backend="cpu"):
    """Initialize the model on a specific device with proper mesh configuration"""
    from easydel import AutoEasyDeLModelForCausalLM
    
    print(f"Creating model with 2D mesh on {backend} backend...")
    
    # Get the appropriate devices based on backend
    if backend == "cpu":
        devices = jax.devices("cpu")
    elif backend == "tt":
        devices = jax.devices("tt")
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    print(f"Using devices: {devices}")
    
    # Create mesh with the specific backend devices
    mesh = Mesh(np.array(devices[0]).reshape(1, 1), axis_names=("x", "y"))
    
    # Define explicit 2D partition rules that work with both TT and CPU
    partition_rules = (
        # Embedding layers - shard columns across tp
        (r"wte/embedding", PartitionSpec("x", "y")),
        (r"wpe/embedding", PartitionSpec()),  # Replicated
        
        # Attention layers
        (r"(attn|crossattention)/c_attn/kernel", PartitionSpec("x", "y")),  # Column-wise
        (r"(attn|crossattention)/q_attn/kernel", PartitionSpec("x", "y")),  # Column-wise
        (r"(attn|crossattention)/c_proj/kernel", PartitionSpec("y", "x")),  # Row-wise

        # MLP layers
        (r"mlp/c_fc/kernel", PartitionSpec("x", "y")),    # Column-wise (input to hidden)
        (r"mlp/c_proj/kernel", PartitionSpec("y", "x")),  # Row-wise (hidden to output)
        
        # Layer norm and biases - replicated
        (r".*/(ln_1|ln_2|ln_cross_attn|ln_f)/scale", PartitionSpec()),
        (r".*/(ln_1|ln_2|ln_cross_attn|ln_f)/bias", PartitionSpec()),
        (r".*(c_attn|q_attn|c_proj|c_fc|lm_head)/bias", PartitionSpec()),
        
        # Language modeling head
        (r"lm_head/kernel", PartitionSpec("x", "y")),  # Column-wise

        # Default fallback - replicated
        (r".*", PartitionSpec()),
    )

    print("Init model")
    # Enter the mesh context BEFORE creating the model
    with mesh:
        model = AutoEasyDeLModelForCausalLM.from_pretrained(
            model_name,
            sharding_axis_dims=(1,1),            # Single device configuration
            sharding_axis_names=("x","y"),         # 2D axis names
            partition_rules=partition_rules,    # Use explicit partition rules
            backend=backend,                    # Specify backend
            verbose=True
        )
        # model = AutoEasyDeLModelForCausalLM.from_pretrained(
        #     model_name,
        #     backend=backend,                    # Specify backend
        #     verbose=True
        # )
    
    # Put model in evaluation mode to disable dropout
    model.eval()
    print(f"Model created successfully with mesh: {model.mesh}")
    print(f"Mesh devices: {model.mesh.devices}")
    
    return model


tt_model = create_model_on_device("gpt2", backend="cpu")

graphdef, state = nnx.split(tt_model)

@jax.jit
def forward_jit(state, inputs):
    model_ = nnx.merge(graphdef, state)
    return model_(inputs)

tokenizer = AutoTokenizer.from_pretrained("gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
prompt = "Today is a beautiful day, and I want to"
tokens = tokenizer(prompt, return_tensors="np", max_length=32, truncation=True)

input_ids = jnp.array(tokens.input_ids)
input_ids_tt = jax.device_put(input_ids, NamedSharding(tt_model.mesh, PartitionSpec()))
print("Trying to run inference on TT model...")
with tt_model.mesh:
        #tt_output = tt_model(input_ids_tt)
        tt_output = forward_jit(state, input_ids_tt)
        print(tt_output)

print("TT model inference runned successfully.")

target_mesh = Mesh(np.array(jax.devices("tt")).reshape(1, 1), axis_names=("x", "y"))
tt_model.config.set_model_mesh(target_mesh)


#graphdef, state = nnx.split(tt_model)
    

state_on_target = jax.tree.map(
    lambda x: jax.device_put(x, NamedSharding(tt_model.mesh, PartitionSpec())),
    state
)


#model_on_target = nnx.merge(graphdef, state_on_target)



input_ids_cpu = jax.device_put(input_ids, NamedSharding(tt_model.mesh, PartitionSpec()))
#cpu_model = create_model_on_device("gpt2", backend="cpu")

with tt_model.mesh:
    #cpu_output = model_on_target(input_ids_cpu)
    cpu_output = forward_jit(state_on_target, input_ids_cpu)



print("CPU model inference runned successfully.")


# Transfer outputs to CPU for comparison
tt_output_cpu = jax.device_put(tt_output, jax.devices("cpu")[0])
cpu_output_cpu = jax.device_put(cpu_output, jax.devices("cpu")[0])

# Calculate PCC (Pearson Correlation Coefficient)
def compute_pcc(x: jax.Array, y: jax.Array):
    """Compute PCC between two tensors"""
    x_flat, y_flat = x.flatten(), y.flatten()
    vx, vy = x_flat - jnp.mean(x_flat), y_flat - jnp.mean(y_flat)
    denom = jnp.linalg.norm(vx) * jnp.linalg.norm(vy)
    return jnp.nan if denom == 0 else jnp.dot(vx, vy) / denom

# Apply PCC calculation to all outputs (handles PyTree structures)
leaf_pccs = jax.tree.map(compute_pcc, tt_output_cpu, cpu_output_cpu)
flat_pccs, _ = jax.tree_util.tree_flatten(leaf_pccs)
pcc = min(flat_pccs)  # Take minimum PCC across all outputs

print(f"\n{'='*60}")
print(f"PCC Comparison Results:")
print(f"{'='*60}")
print(f"Minimum PCC: {pcc:.6f}")
print(f"All PCCs: {flat_pccs}")
print(f"Status: {'✓ PASS' if pcc >= 0.99 else '✗ FAIL'} (threshold: 0.99)")
print(f"{'='*60}\n")