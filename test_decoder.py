# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.utils.quantization_config import Mxfp4Config


def setup_spmd():
    """Initialize SPMD mode in torch_xla."""
    print("Setting up XLA SPMD environment...")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    print("XLA SPMD environment configured.")


def create_device_mesh() -> Mesh:
    """Create device mesh for tensor parallelism."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def mark_sharding_on_layer(layer: torch.nn.Module, mesh: Mesh):
    """Apply tensor parallel sharding to a single GPT-OSS layer."""
    # Attention layer sharding
    xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(layer.self_attn.q_proj.bias, mesh, ("model",))
    xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(layer.self_attn.k_proj.bias, mesh, ("model",))
    xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(layer.self_attn.v_proj.bias, mesh, ("model",))
    xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))
    xs.mark_sharding(layer.self_attn.o_proj.bias, mesh, ("batch",))
    xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

    # MLP layer sharding
    xs.mark_sharding(layer.mlp.router.weight, mesh, (None, "batch"))
    xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", "batch", None))
    xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
    xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, "batch"))
    xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", "batch"))

    # Layer normalization
    xs.mark_sharding(layer.input_layernorm.weight, mesh, ("batch",))
    xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, ("batch",))


def compute_pcc(tensor1, tensor2):
    """Compute Pearson Correlation Coefficient between two tensors."""
    # Flatten tensors
    t1_flat = tensor1.flatten().float()
    t2_flat = tensor2.flatten().float()

    # Compute PCC
    mean1 = t1_flat.mean()
    mean2 = t2_flat.mean()

    numerator = ((t1_flat - mean1) * (t2_flat - mean2)).sum()
    denominator = torch.sqrt(
        ((t1_flat - mean1) ** 2).sum() * ((t2_flat - mean2) ** 2).sum()
    )

    pcc = numerator / denominator
    return pcc.item()


def test_single_layer_multiple_runs():
    """
    Test each layer sequentially on both CPU and TT device.
    Compares outputs layer-by-layer and saves hidden states.
    """
    setup_spmd()

    # Create directories for saving hidden states
    os.makedirs("cpu_hidden_states", exist_ok=True)
    os.makedirs("tt_hidden_states", exist_ok=True)

    # Connect the device and create an xla mesh
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Load config and model
    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Load captured inputs
    print("\nLoading captured inputs from 'first_layer_inputs.pt'...")
    captured_inputs = torch.load("first_layer_inputs.pt")
    print(f"Loaded hidden states shape: {captured_inputs['hidden_states'].shape}")

    # Prepare inputs
    hidden_states_cpu = captured_inputs["hidden_states"]
    attention_mask = captured_inputs["attention_mask"]
    position_ids = captured_inputs["position_ids"]
    cache_position = captured_inputs["cache_position"]
    position_embeddings = captured_inputs["position_embeddings"]

    # num_layers = len(model.model.layers)
    num_layers = 13
    pccs = []

    # ========================================
    # Run all layers on CPU sequentially
    # ========================================
    print(f"\n{'='*60}")
    print("Running all layers on CPU (reference)...")
    print(f"{'='*60}")

    cpu_hidden_states = []
    current_hidden_states = hidden_states_cpu.clone()

    for layer_idx in range(num_layers):
        print(f"\nCPU - Layer {layer_idx}/{num_layers}...")

        layer = model.model.layers[layer_idx].cpu()

        with torch.no_grad():
            layer_output = layer(
                current_hidden_states,
                attention_mask=(
                    attention_mask
                    if not isinstance(attention_mask, dict)
                    else {
                        k: v if isinstance(v, torch.Tensor) else v
                        for k, v in attention_mask.items()
                    }
                ),
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            # Extract hidden states (handle both tuple and tensor return types)
            if isinstance(layer_output, tuple):
                current_hidden_states = layer_output[0]
            else:
                current_hidden_states = layer_output

        cpu_hidden_states.append(current_hidden_states.clone())

        # Save CPU hidden states
        torch.save(current_hidden_states, f"cpu_hidden_states/layer_{layer_idx}.pt")
        print(f"CPU Layer {layer_idx} output shape: {current_hidden_states.shape}")

    # ========================================
    # Run all layers on TT device sequentially
    # ========================================
    print(f"\n{'='*60}")
    print("Running all layers on TT device...")
    print(f"{'='*60}")

    tt_hidden_states = []
    current_hidden_states = hidden_states_cpu.clone().to(device)

    # Prepare inputs for TT device
    attention_mask_tt = attention_mask
    if isinstance(attention_mask, dict):
        attention_mask_tt = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in attention_mask.items()
        }
    elif attention_mask is not None:
        attention_mask_tt = attention_mask.to(device)

    position_ids_tt = position_ids.to(device) if position_ids is not None else None
    cache_position_tt = (
        cache_position.to(device) if cache_position is not None else None
    )

    if position_embeddings is not None:
        cos, sin = position_embeddings
        position_embeddings_tt = (cos.to(device), sin.to(device))
    else:
        position_embeddings_tt = None

    for layer_idx in range(num_layers):
        print(f"\nTT - Layer {layer_idx}/{num_layers}...")

        # Get layer and move to TT device
        layer = model.model.layers[layer_idx].to(device)

        # Apply sharding
        print(f"Applying sharding to layer {layer_idx}...")
        mark_sharding_on_layer(layer, mesh)

        # Compile the layer
        print(f"Compiling layer {layer_idx} with torch.compile backend='tt'...")
        compiled_layer = torch.compile(layer, backend="tt")

        # Run on TT device
        with torch.no_grad():
            layer_output = compiled_layer(
                current_hidden_states,
                attention_mask=attention_mask_tt,
                position_ids=position_ids_tt,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position_tt,
                position_embeddings=position_embeddings_tt,
            )
            # Extract hidden states (handle both tuple and tensor return types)
            if isinstance(layer_output, tuple):
                current_hidden_states = layer_output[0]
            else:
                current_hidden_states = layer_output

        # Move to CPU for comparison and saving
        current_hidden_states_cpu = current_hidden_states.cpu()
        tt_hidden_states.append(current_hidden_states_cpu.clone())

        # Save TT hidden states
        torch.save(current_hidden_states_cpu, f"tt_hidden_states/layer_{layer_idx}.pt")
        print(f"TT Layer {layer_idx} output shape: {current_hidden_states.shape}")

        # Compute PCC against CPU reference for this layer
        pcc = compute_pcc(cpu_hidden_states[layer_idx], current_hidden_states_cpu)
        pccs.append(pcc)

        print(f"Layer {layer_idx} PCC (TT vs CPU): {pcc:.6f}")

    # ========================================
    # Summary
    # ========================================
    torch.save(pccs, "layer_pccs.pt")
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"All layer PCCs saved to 'layer_pccs.pt'")
    print(f"CPU hidden states saved to 'cpu_hidden_states/' directory")
    print(f"TT hidden states saved to 'tt_hidden_states/' directory")
    print(f"Average PCC: {np.mean(pccs):.6f}")
    print(f"Min PCC: {np.min(pccs):.6f} at layer {np.argmin(pccs)}")
    print(f"Max PCC: {np.max(pccs):.6f} at layer {np.argmax(pccs)}")
    print(f"PCC std dev: {np.std(pccs):.6f}")


def test_specific_layer(layer_idx: int):
    """
    Test a specific layer by loading saved CPU hidden states.

    Args:
        layer_idx: The layer index to test (0-based)
    """
    setup_spmd()

    # Create directory for saving outputs
    os.makedirs("specific_layer_outputs", exist_ok=True)

    # Check if the required CPU hidden states file exists
    if layer_idx == 0:
        # For layer 0, use the original input
        input_file = "first_layer_inputs.pt"
    else:
        # For other layers, use the output from the previous layer
        input_file = f"cpu_hidden_states/layer_{layer_idx - 1}.pt"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(
            f"Please run test_single_layer_multiple_runs() first to generate hidden states,"
        )
        print(f"or provide the correct input file.")
        return

    print(f"\n{'='*60}")
    print(f"Testing Layer {layer_idx}")
    print(f"{'='*60}")

    # Connect the device and create an xla mesh
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Load config and model
    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Check if layer index is valid
    if layer_idx >= len(model.model.layers):
        print(f"Error: Layer index {layer_idx} is out of range!")
        print(
            f"Model has {len(model.model.layers)} layers (indices 0-{len(model.model.layers)-1})"
        )
        return

    # Load input hidden states
    if layer_idx == 0:
        print(f"\nLoading initial inputs from '{input_file}'...")
        captured_inputs = torch.load(input_file)
        hidden_states_cpu = captured_inputs["hidden_states"]
        attention_mask = captured_inputs["attention_mask"]
        position_ids = captured_inputs["position_ids"]
        cache_position = captured_inputs["cache_position"]
        position_embeddings = captured_inputs["position_embeddings"]
    else:
        print(f"\nLoading hidden states from '{input_file}'...")
        hidden_states_cpu = torch.load(input_file)

        # Load the original inputs for other parameters
        print(
            "Loading attention mask and position info from 'first_layer_inputs.pt'..."
        )
        captured_inputs = torch.load("first_layer_inputs.pt")
        attention_mask = captured_inputs["attention_mask"]
        position_ids = captured_inputs["position_ids"]
        cache_position = captured_inputs["cache_position"]
        position_embeddings = captured_inputs["position_embeddings"]

    print(f"Input hidden states shape: {hidden_states_cpu.shape}")

    # Get the specific layer
    layer = model.model.layers[layer_idx]

    # ========================================
    # Run on CPU
    # ========================================
    print(f"\nRunning Layer {layer_idx} on CPU...")
    layer_cpu = layer.cpu()

    with torch.no_grad():
        layer_output = layer_cpu(
            hidden_states_cpu,
            attention_mask=(
                attention_mask
                if not isinstance(attention_mask, dict)
                else {
                    k: v if isinstance(v, torch.Tensor) else v
                    for k, v in attention_mask.items()
                }
            ),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        # Extract hidden states (handle both tuple and tensor return types)
        if isinstance(layer_output, tuple):
            cpu_output = layer_output[0]
        else:
            cpu_output = layer_output

    print(f"CPU output shape: {cpu_output.shape}")

    # Save CPU output
    cpu_output_file = f"specific_layer_outputs/layer_{layer_idx}_cpu.pt"
    torch.save(cpu_output, cpu_output_file)
    print(f"Saved CPU output to '{cpu_output_file}'")

    # ========================================
    # Run on TT device
    # ========================================
    print(f"\nRunning Layer {layer_idx} on TT device...")

    # Move layer to TT device
    layer_tt = layer.to(device)

    # Apply sharding
    print(f"Applying sharding to layer {layer_idx}...")
    mark_sharding_on_layer(layer_tt, mesh)

    # Compile the layer
    print(f"Compiling layer {layer_idx} with torch.compile backend='tt'...")
    compiled_layer = torch.compile(layer_tt, backend="tt")

    # Prepare inputs for TT device
    hidden_states_tt = hidden_states_cpu.to(device)

    attention_mask_tt = attention_mask
    if isinstance(attention_mask, dict):
        attention_mask_tt = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in attention_mask.items()
        }
    elif attention_mask is not None:
        attention_mask_tt = attention_mask.to(device)

    position_ids_tt = position_ids.to(device) if position_ids is not None else None
    cache_position_tt = (
        cache_position.to(device) if cache_position is not None else None
    )

    if position_embeddings is not None:
        cos, sin = position_embeddings
        position_embeddings_tt = (cos.to(device), sin.to(device))
    else:
        position_embeddings_tt = None

    # Run on TT device
    with torch.no_grad():
        layer_output = compiled_layer(
            hidden_states_tt,
            attention_mask=attention_mask_tt,
            position_ids=position_ids_tt,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position_tt,
            position_embeddings=position_embeddings_tt,
        )
        # Extract hidden states (handle both tuple and tensor return types)
        if isinstance(layer_output, tuple):
            tt_output = layer_output[0]
        else:
            tt_output = layer_output

    # Move to CPU for comparison and saving
    tt_output_cpu = tt_output.cpu()
    print(f"TT output shape: {tt_output.shape}")

    # Save TT output
    tt_output_file = f"specific_layer_outputs/layer_{layer_idx}_tt.pt"
    torch.save(tt_output_cpu, tt_output_file)
    print(f"Saved TT output to '{tt_output_file}'")

    # ========================================
    # Compute PCC
    # ========================================
    pcc = compute_pcc(cpu_output, tt_output_cpu)

    # Save PCC
    pcc_file = f"specific_layer_outputs/layer_{layer_idx}_pcc.pt"
    torch.save({"layer_idx": layer_idx, "pcc": pcc}, pcc_file)

    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} Results")
    print(f"{'='*60}")
    print(f"PCC (TT vs CPU): {pcc:.6f}")
    print(f"PCC saved to '{pcc_file}'")
    print(f"CPU output saved to '{cpu_output_file}'")
    print(f"TT output saved to '{tt_output_file}'")
    print(f"{'='*60}")

    return pcc


def test_layer_components(layer_idx: int):
    """
    Test individual components of a specific layer separately.
    Compiles and tests: input_layernorm, self_attn, post_attention_layernorm, mlp

    Args:
        layer_idx: The layer index to test (0-based)
    """
    setup_spmd()

    # Create directory for saving outputs
    os.makedirs("component_outputs", exist_ok=True)

    # Check if the required CPU hidden states file exists
    if layer_idx == 0:
        input_file = "first_layer_inputs.pt"
    else:
        input_file = f"cpu_hidden_states/layer_{layer_idx - 1}.pt"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(
            f"Please run test_single_layer_multiple_runs() first to generate hidden states."
        )
        return

    print(f"\n{'='*60}")
    print(f"Testing Layer {layer_idx} Components")
    print(f"{'='*60}")

    # Connect the device and create an xla mesh
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Load config and model
    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Check if layer index is valid
    if layer_idx >= len(model.model.layers):
        print(f"Error: Layer index {layer_idx} is out of range!")
        print(
            f"Model has {len(model.model.layers)} layers (indices 0-{len(model.model.layers)-1})"
        )
        return

    # Load input hidden states
    if layer_idx == 0:
        print(f"\nLoading initial inputs from '{input_file}'...")
        captured_inputs = torch.load(input_file)
        hidden_states_cpu = captured_inputs["hidden_states"]
        attention_mask = captured_inputs["attention_mask"]
        position_ids = captured_inputs["position_ids"]
        cache_position = captured_inputs["cache_position"]
        position_embeddings = captured_inputs["position_embeddings"]
    else:
        print(f"\nLoading hidden states from '{input_file}'...")
        hidden_states_cpu = torch.load(input_file)

        print(
            "Loading attention mask and position info from 'first_layer_inputs.pt'..."
        )
        captured_inputs = torch.load("first_layer_inputs.pt")
        attention_mask = captured_inputs["attention_mask"]
        position_ids = captured_inputs["position_ids"]
        cache_position = captured_inputs["cache_position"]
        position_embeddings = captured_inputs["position_embeddings"]

    print(f"Input hidden states shape: {hidden_states_cpu.shape}")

    # Get the specific layer
    layer = model.model.layers[layer_idx]

    # Dictionary to store PCC results
    pcc_results = {}

    # ========================================
    # Component 1: input_layernorm
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 1: input_layernorm")
    print(f"{'='*60}")

    # CPU reference
    print("Running input_layernorm on CPU...")
    input_ln_cpu = layer.input_layernorm.cpu()
    with torch.no_grad():
        norm1_output_cpu = input_ln_cpu(hidden_states_cpu)
    print(f"CPU output shape: {norm1_output_cpu.shape}")

    # TT device
    print("Running input_layernorm on TT device...")
    input_ln_tt = layer.input_layernorm.to(device)
    xs.mark_sharding(input_ln_tt.weight, mesh, ("batch",))
    compiled_input_ln = torch.compile(input_ln_tt, backend="tt")

    hidden_states_tt = hidden_states_cpu.to(device)
    with torch.no_grad():
        norm1_output_tt = compiled_input_ln(hidden_states_tt)
    norm1_output_tt_cpu = norm1_output_tt.cpu()
    print(f"TT output shape: {norm1_output_tt.shape}")

    # Compute PCC
    pcc_norm1 = compute_pcc(norm1_output_cpu, norm1_output_tt_cpu)
    pcc_results["input_layernorm"] = pcc_norm1
    print(f"input_layernorm PCC: {pcc_norm1:.6f}")

    # Save outputs
    torch.save(norm1_output_cpu, f"component_outputs/layer_{layer_idx}_input_ln_cpu.pt")
    torch.save(
        norm1_output_tt_cpu, f"component_outputs/layer_{layer_idx}_input_ln_tt.pt"
    )

    # ========================================
    # Component 2: self_attn
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 2: self_attn")
    print(f"{'='*60}")

    # CPU reference
    print("Running self_attn on CPU...")
    self_attn_cpu = layer.self_attn.cpu()
    with torch.no_grad():
        attn_output_cpu, _ = self_attn_cpu(
            hidden_states=norm1_output_cpu,
            attention_mask=(
                attention_mask
                if not isinstance(attention_mask, dict)
                else {
                    k: v if isinstance(v, torch.Tensor) else v
                    for k, v in attention_mask.items()
                }
            ),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    print(f"CPU output shape: {attn_output_cpu.shape}")

    # TT device
    print("Running self_attn on TT device...")
    self_attn_tt = layer.self_attn.to(device)

    # Apply sharding to attention
    xs.mark_sharding(self_attn_tt.q_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(self_attn_tt.q_proj.bias, mesh, ("model",))
    xs.mark_sharding(self_attn_tt.k_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(self_attn_tt.k_proj.bias, mesh, ("model",))
    xs.mark_sharding(self_attn_tt.v_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(self_attn_tt.v_proj.bias, mesh, ("model",))
    xs.mark_sharding(self_attn_tt.o_proj.weight, mesh, ("batch", "model"))
    xs.mark_sharding(self_attn_tt.o_proj.bias, mesh, ("batch",))
    xs.mark_sharding(self_attn_tt.sinks, mesh, (None,))

    compiled_self_attn = torch.compile(self_attn_tt, backend="tt")

    # Prepare inputs for TT device
    norm1_output_device = norm1_output_tt  # Already on device from previous step

    attention_mask_tt = attention_mask
    if isinstance(attention_mask, dict):
        attention_mask_tt = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in attention_mask.items()
        }
    elif attention_mask is not None:
        attention_mask_tt = attention_mask.to(device)

    position_ids_tt = position_ids.to(device) if position_ids is not None else None
    cache_position_tt = (
        cache_position.to(device) if cache_position is not None else None
    )

    if position_embeddings is not None:
        cos, sin = position_embeddings
        position_embeddings_tt = (cos.to(device), sin.to(device))
    else:
        position_embeddings_tt = None

    with torch.no_grad():
        attn_output_tt, _ = compiled_self_attn(
            hidden_states=norm1_output_device,
            attention_mask=attention_mask_tt,
            position_ids=position_ids_tt,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position_tt,
            position_embeddings=position_embeddings_tt,
        )
    attn_output_tt_cpu = attn_output_tt.cpu()
    print(f"TT output shape: {attn_output_tt.shape}")

    # Compute PCC
    pcc_attn = compute_pcc(attn_output_cpu, attn_output_tt_cpu)
    pcc_results["self_attn"] = pcc_attn
    print(f"self_attn PCC: {pcc_attn:.6f}")

    # Save outputs
    torch.save(attn_output_cpu, f"component_outputs/layer_{layer_idx}_attn_cpu.pt")
    torch.save(attn_output_tt_cpu, f"component_outputs/layer_{layer_idx}_attn_tt.pt")

    # ========================================
    # First Residual Connection (CPU reference)
    # ========================================
    print("\nApplying first residual connection...")
    residual1_cpu = hidden_states_cpu + attn_output_cpu
    residual1_tt = hidden_states_tt + attn_output_tt
    residual1_tt_cpu = residual1_tt.cpu()

    pcc_residual1 = compute_pcc(residual1_cpu, residual1_tt_cpu)
    pcc_results["first_residual"] = pcc_residual1
    print(f"First residual PCC: {pcc_residual1:.6f}")

    # ========================================
    # Component 3: post_attention_layernorm
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 3: post_attention_layernorm")
    print(f"{'='*60}")

    # CPU reference
    print("Running post_attention_layernorm on CPU...")
    post_attn_ln_cpu = layer.post_attention_layernorm.cpu()
    with torch.no_grad():
        norm2_output_cpu = post_attn_ln_cpu(residual1_cpu)
    print(f"CPU output shape: {norm2_output_cpu.shape}")

    # TT device
    print("Running post_attention_layernorm on TT device...")
    post_attn_ln_tt = layer.post_attention_layernorm.to(device)
    xs.mark_sharding(post_attn_ln_tt.weight, mesh, ("batch",))
    compiled_post_attn_ln = torch.compile(post_attn_ln_tt, backend="tt")

    with torch.no_grad():
        norm2_output_tt = compiled_post_attn_ln(residual1_tt)
    norm2_output_tt_cpu = norm2_output_tt.cpu()
    print(f"TT output shape: {norm2_output_tt.shape}")

    # Compute PCC
    pcc_norm2 = compute_pcc(norm2_output_cpu, norm2_output_tt_cpu)
    pcc_results["post_attention_layernorm"] = pcc_norm2
    print(f"post_attention_layernorm PCC: {pcc_norm2:.6f}")

    # Save outputs
    torch.save(
        norm2_output_cpu, f"component_outputs/layer_{layer_idx}_post_attn_ln_cpu.pt"
    )
    torch.save(
        norm2_output_tt_cpu, f"component_outputs/layer_{layer_idx}_post_attn_ln_tt.pt"
    )

    # ========================================
    # Component 4: mlp
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 4: mlp")
    print(f"{'='*60}")

    # CPU reference
    print("Running mlp on CPU...")
    mlp_cpu = layer.mlp.cpu()
    with torch.no_grad():
        mlp_output_cpu, _ = mlp_cpu(norm2_output_cpu)
    print(f"CPU output shape: {mlp_output_cpu.shape}")

    # TT device
    print("Running mlp on TT device...")
    mlp_tt = layer.mlp.to(device)

    # Apply sharding to MLP
    xs.mark_sharding(mlp_tt.router.weight, mesh, (None, "batch"))
    xs.mark_sharding(mlp_tt.experts.gate_up_proj, mesh, ("model", "batch", None))
    xs.mark_sharding(mlp_tt.experts.gate_up_proj_bias, mesh, ("model", None))
    xs.mark_sharding(mlp_tt.experts.down_proj, mesh, ("model", None, "batch"))
    xs.mark_sharding(mlp_tt.experts.down_proj_bias, mesh, ("model", "batch"))

    compiled_mlp = torch.compile(mlp_tt, backend="tt")

    with torch.no_grad():
        mlp_output_tt, _ = compiled_mlp(norm2_output_tt)
    mlp_output_tt_cpu = mlp_output_tt.cpu()
    print(f"TT output shape: {mlp_output_tt.shape}")

    # Compute PCC
    pcc_mlp = compute_pcc(mlp_output_cpu, mlp_output_tt_cpu)
    pcc_results["mlp"] = pcc_mlp
    print(f"mlp PCC: {pcc_mlp:.6f}")

    # Save outputs
    torch.save(mlp_output_cpu, f"component_outputs/layer_{layer_idx}_mlp_cpu.pt")
    torch.save(mlp_output_tt_cpu, f"component_outputs/layer_{layer_idx}_mlp_tt.pt")

    # ========================================
    # Second Residual Connection (CPU reference)
    # ========================================
    print("\nApplying second residual connection...")
    final_output_cpu = residual1_cpu + mlp_output_cpu
    final_output_tt = residual1_tt + mlp_output_tt
    final_output_tt_cpu = final_output_tt.cpu()

    pcc_residual2 = compute_pcc(final_output_cpu, final_output_tt_cpu)
    pcc_results["second_residual"] = pcc_residual2
    print(f"Second residual PCC: {pcc_residual2:.6f}")

    pcc_final = compute_pcc(final_output_cpu, final_output_tt_cpu)
    pcc_results["final_output"] = pcc_final
    print(f"Final layer output PCC: {pcc_final:.6f}")

    # Save final outputs
    torch.save(final_output_cpu, f"component_outputs/layer_{layer_idx}_final_cpu.pt")
    torch.save(final_output_tt_cpu, f"component_outputs/layer_{layer_idx}_final_tt.pt")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} Component Test Summary")
    print(f"{'='*60}")
    for component, pcc in pcc_results.items():
        print(f"{component:30s}: PCC = {pcc:.6f}")

    # Save PCC results
    pcc_file = f"component_outputs/layer_{layer_idx}_component_pccs.pt"
    torch.save({"layer_idx": layer_idx, "pccs": pcc_results}, pcc_file)
    print(f"\nPCC results saved to '{pcc_file}'")
    print(f"All component outputs saved to 'component_outputs/' directory")
    print(f"{'='*60}")

    return pcc_results


def test_layer_components_from_cpu_output(layer_idx: int):
    """
    Test individual components of a specific layer separately.
    Compiles and tests: input_layernorm, self_attn, post_attention_layernorm, mlp
    Each component uses CPU reference inputs to isolate errors.

    Args:
        layer_idx: The layer index to test (0-based)
    """
    setup_spmd()

    # Create directory for saving outputs
    os.makedirs("component_outputs", exist_ok=True)

    # Check if the required CPU hidden states file exists
    if layer_idx == 0:
        input_file = "first_layer_inputs.pt"
    else:
        input_file = f"cpu_hidden_states/layer_{layer_idx - 1}.pt"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(
            f"Please run test_single_layer_multiple_runs() first to generate hidden states."
        )
        return

    print(f"\n{'='*60}")
    print(f"Testing Layer {layer_idx} Components")
    print(f"{'='*60}")

    # Connect the device and create an xla mesh
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Load config and model
    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Check if layer index is valid
    if layer_idx >= len(model.model.layers):
        print(f"Error: Layer index {layer_idx} is out of range!")
        print(
            f"Model has {len(model.model.layers)} layers (indices 0-{len(model.model.layers)-1})"
        )
        return

    # Load input hidden states
    if layer_idx == 0:
        print(f"\nLoading initial inputs from '{input_file}'...")
        captured_inputs = torch.load(input_file)
        hidden_states_cpu = captured_inputs["hidden_states"]
        attention_mask = captured_inputs["attention_mask"]
        position_ids = captured_inputs["position_ids"]
        cache_position = captured_inputs["cache_position"]
        position_embeddings = captured_inputs["position_embeddings"]
    else:
        print(f"\nLoading hidden states from '{input_file}'...")
        hidden_states_cpu = torch.load(input_file)

        print(
            "Loading attention mask and position info from 'first_layer_inputs.pt'..."
        )
        captured_inputs = torch.load("first_layer_inputs.pt")
        attention_mask = captured_inputs["attention_mask"]
        position_ids = captured_inputs["position_ids"]
        cache_position = captured_inputs["cache_position"]
        position_embeddings = captured_inputs["position_embeddings"]

    print(f"Input hidden states shape: {hidden_states_cpu.shape}")

    # Get the specific layer
    layer = model.model.layers[layer_idx]

    # Dictionary to store PCC results
    pcc_results = {}

    # ========================================
    # Component 1: input_layernorm
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 1: input_layernorm")
    print(f"{'='*60}")

    # CPU reference
    print("Running input_layernorm on CPU...")
    input_ln_cpu = layer.input_layernorm.cpu()
    with torch.no_grad():
        norm1_output_cpu = input_ln_cpu(hidden_states_cpu)
    print(f"CPU output shape: {norm1_output_cpu.shape}")

    # TT device - uses CPU input
    print("Running input_layernorm on TT device...")
    input_ln_tt = layer.input_layernorm.to(device)
    xs.mark_sharding(input_ln_tt.weight, mesh, ("batch",))
    compiled_input_ln = torch.compile(input_ln_tt, backend="tt")

    hidden_states_tt = hidden_states_cpu.to(device)
    with torch.no_grad():
        norm1_output_tt = compiled_input_ln(hidden_states_tt)
    norm1_output_tt_cpu = norm1_output_tt.cpu()
    print(f"TT output shape: {norm1_output_tt.shape}")

    # Compute PCC
    pcc_norm1 = compute_pcc(norm1_output_cpu, norm1_output_tt_cpu)
    pcc_results["input_layernorm"] = pcc_norm1
    print(f"input_layernorm PCC: {pcc_norm1:.6f}")

    # Save outputs
    torch.save(norm1_output_cpu, f"component_outputs/layer_{layer_idx}_input_ln_cpu.pt")
    torch.save(
        norm1_output_tt_cpu, f"component_outputs/layer_{layer_idx}_input_ln_tt.pt"
    )

    # ========================================
    # Component 2: self_attn
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 2: self_attn")
    print(f"{'='*60}")

    # CPU reference
    print("Running self_attn on CPU...")
    self_attn_cpu = layer.self_attn.cpu()
    with torch.no_grad():
        attn_output_cpu, _ = self_attn_cpu(
            hidden_states=norm1_output_cpu,
            attention_mask=(
                attention_mask
                if not isinstance(attention_mask, dict)
                else {
                    k: v if isinstance(v, torch.Tensor) else v
                    for k, v in attention_mask.items()
                }
            ),
            position_ids=position_ids,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
    print(f"CPU output shape: {attn_output_cpu.shape}")

    # TT device - uses CPU output from norm1 (not TT output)
    print("Running self_attn on TT device...")
    self_attn_tt = layer.self_attn.to(device)

    # Apply sharding to attention
    xs.mark_sharding(self_attn_tt.q_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(self_attn_tt.q_proj.bias, mesh, ("model",))
    xs.mark_sharding(self_attn_tt.k_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(self_attn_tt.k_proj.bias, mesh, ("model",))
    xs.mark_sharding(self_attn_tt.v_proj.weight, mesh, ("model", "batch"))
    xs.mark_sharding(self_attn_tt.v_proj.bias, mesh, ("model",))
    xs.mark_sharding(self_attn_tt.o_proj.weight, mesh, ("batch", "model"))
    xs.mark_sharding(self_attn_tt.o_proj.bias, mesh, ("batch",))
    xs.mark_sharding(self_attn_tt.sinks, mesh, (None,))

    compiled_self_attn = torch.compile(self_attn_tt, backend="tt")

    # Use CPU reference output from norm1, not TT output
    norm1_output_device = norm1_output_cpu.to(device)

    attention_mask_tt = attention_mask
    if isinstance(attention_mask, dict):
        attention_mask_tt = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in attention_mask.items()
        }
    elif attention_mask is not None:
        attention_mask_tt = attention_mask.to(device)

    position_ids_tt = position_ids.to(device) if position_ids is not None else None
    cache_position_tt = (
        cache_position.to(device) if cache_position is not None else None
    )

    if position_embeddings is not None:
        cos, sin = position_embeddings
        position_embeddings_tt = (cos.to(device), sin.to(device))
    else:
        position_embeddings_tt = None

    with torch.no_grad():
        attn_output_tt, _ = compiled_self_attn(
            hidden_states=norm1_output_device,
            attention_mask=attention_mask_tt,
            position_ids=position_ids_tt,
            past_key_values=None,
            use_cache=False,
            cache_position=cache_position_tt,
            position_embeddings=position_embeddings_tt,
        )
    attn_output_tt_cpu = attn_output_tt.cpu()
    print(f"TT output shape: {attn_output_tt.shape}")

    # Compute PCC
    pcc_attn = compute_pcc(attn_output_cpu, attn_output_tt_cpu)
    pcc_results["self_attn"] = pcc_attn
    print(f"self_attn PCC: {pcc_attn:.6f}")

    # Save outputs
    torch.save(attn_output_cpu, f"component_outputs/layer_{layer_idx}_attn_cpu.pt")
    torch.save(attn_output_tt_cpu, f"component_outputs/layer_{layer_idx}_attn_tt.pt")

    # ========================================
    # First Residual Connection (CPU reference)
    # ========================================
    print("\nApplying first residual connection...")
    residual1_cpu = hidden_states_cpu + attn_output_cpu
    residual1_tt = hidden_states_tt + attn_output_tt
    residual1_tt_cpu = residual1_tt.cpu()

    pcc_residual1 = compute_pcc(residual1_cpu, residual1_tt_cpu)
    pcc_results["first_residual"] = pcc_residual1
    print(f"First residual PCC: {pcc_residual1:.6f}")

    # ========================================
    # Component 3: post_attention_layernorm
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 3: post_attention_layernorm")
    print(f"{'='*60}")

    # CPU reference
    print("Running post_attention_layernorm on CPU...")
    post_attn_ln_cpu = layer.post_attention_layernorm.cpu()
    with torch.no_grad():
        norm2_output_cpu = post_attn_ln_cpu(residual1_cpu)
    print(f"CPU output shape: {norm2_output_cpu.shape}")

    # TT device - uses CPU residual1, not TT residual1
    print("Running post_attention_layernorm on TT device...")
    post_attn_ln_tt = layer.post_attention_layernorm.to(device)
    xs.mark_sharding(post_attn_ln_tt.weight, mesh, ("batch",))
    compiled_post_attn_ln = torch.compile(post_attn_ln_tt, backend="tt")

    # Use CPU reference residual, not TT residual
    residual1_device = residual1_cpu.to(device)
    with torch.no_grad():
        norm2_output_tt = compiled_post_attn_ln(residual1_device)
    norm2_output_tt_cpu = norm2_output_tt.cpu()
    print(f"TT output shape: {norm2_output_tt.shape}")

    # Compute PCC
    pcc_norm2 = compute_pcc(norm2_output_cpu, norm2_output_tt_cpu)
    pcc_results["post_attention_layernorm"] = pcc_norm2
    print(f"post_attention_layernorm PCC: {pcc_norm2:.6f}")

    # Save outputs
    torch.save(
        norm2_output_cpu, f"component_outputs/layer_{layer_idx}_post_attn_ln_cpu.pt"
    )
    torch.save(
        norm2_output_tt_cpu, f"component_outputs/layer_{layer_idx}_post_attn_ln_tt.pt"
    )

    # ========================================
    # Component 4: mlp
    # ========================================
    print(f"\n{'='*60}")
    print(f"Testing Component 4: mlp")
    print(f"{'='*60}")

    # CPU reference
    print("Running mlp on CPU...")
    mlp_cpu = layer.mlp.cpu()
    with torch.no_grad():
        mlp_output_cpu, _ = mlp_cpu(norm2_output_cpu)
    print(f"CPU output shape: {mlp_output_cpu.shape}")

    # TT device - uses CPU norm2 output, not TT norm2 output
    print("Running mlp on TT device...")
    mlp_tt = layer.mlp.to(device)

    # Apply sharding to MLP
    xs.mark_sharding(mlp_tt.router.weight, mesh, (None, "batch"))
    xs.mark_sharding(mlp_tt.experts.gate_up_proj, mesh, ("model", "batch", None))
    xs.mark_sharding(mlp_tt.experts.gate_up_proj_bias, mesh, ("model", None))
    xs.mark_sharding(mlp_tt.experts.down_proj, mesh, ("model", None, "batch"))
    xs.mark_sharding(mlp_tt.experts.down_proj_bias, mesh, ("model", "batch"))

    compiled_mlp = torch.compile(mlp_tt, backend="tt")

    # Use CPU reference norm2 output, not TT norm2 output
    norm2_output_device = norm2_output_cpu.to(device)
    with torch.no_grad():
        mlp_output_tt, _ = compiled_mlp(norm2_output_device)
    mlp_output_tt_cpu = mlp_output_tt.cpu()
    print(f"TT output shape: {mlp_output_tt.shape}")

    # Compute PCC
    pcc_mlp = compute_pcc(mlp_output_cpu, mlp_output_tt_cpu)
    pcc_results["mlp"] = pcc_mlp
    print(f"mlp PCC: {pcc_mlp:.6f}")

    # Save outputs
    torch.save(mlp_output_cpu, f"component_outputs/layer_{layer_idx}_mlp_cpu.pt")
    torch.save(mlp_output_tt_cpu, f"component_outputs/layer_{layer_idx}_mlp_tt.pt")

    # ========================================
    # Second Residual Connection (CPU reference)
    # ========================================
    print("\nApplying second residual connection...")
    final_output_cpu = residual1_cpu + mlp_output_cpu
    final_output_tt = residual1_tt + mlp_output_tt
    final_output_tt_cpu = final_output_tt.cpu()

    pcc_residual2 = compute_pcc(final_output_cpu, final_output_tt_cpu)
    pcc_results["second_residual"] = pcc_residual2
    print(f"Second residual PCC: {pcc_residual2:.6f}")

    pcc_final = compute_pcc(final_output_cpu, final_output_tt_cpu)
    pcc_results["final_output"] = pcc_final
    print(f"Final layer output PCC: {pcc_final:.6f}")

    # Save final outputs
    torch.save(final_output_cpu, f"component_outputs/layer_{layer_idx}_final_cpu.pt")
    torch.save(final_output_tt_cpu, f"component_outputs/layer_{layer_idx}_final_tt.pt")

    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*60}")
    print(f"Layer {layer_idx} Component Test Summary")
    print(f"{'='*60}")
    for component, pcc in pcc_results.items():
        print(f"{component:30s}: PCC = {pcc:.6f}")

    # Save PCC results
    pcc_file = f"component_outputs/layer_{layer_idx}_component_pccs.pt"
    torch.save({"layer_idx": layer_idx, "pccs": pcc_results}, pcc_file)
    print(f"\nPCC results saved to '{pcc_file}'")
    print(f"All component outputs saved to 'component_outputs/' directory")
    print(f"{'='*60}")

    return pcc_results


def test_input_layernorm(layer_idx: int):
    """
    Deep debug test for input_layernorm of a specific layer.
    Analyzes input statistics, intermediate computations, and output quality.

    Args:
        layer_idx: The layer index to test (0-based)
    """
    setup_spmd()

    # Create directory for saving outputs
    os.makedirs("layernorm_debug", exist_ok=True)

    # Check if the required input file exists
    if layer_idx == 0:
        input_file = "first_layer_inputs.pt"
    else:
        input_file = f"cpu_hidden_states/layer_{layer_idx - 1}.pt"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please run test_single_layer_multiple_runs() first.")
        return

    # Check if component outputs exist for verification
    cpu_output_file = f"component_outputs/layer_{layer_idx}_input_ln_cpu.pt"
    tt_output_file = f"component_outputs/layer_{layer_idx}_input_ln_tt.pt"
    if not os.path.exists(cpu_output_file) or not os.path.exists(tt_output_file):
        print(
            f"Warning: Component output files not found. Run test_layer_components({layer_idx}) first for comparison."
        )

    print(f"\n{'='*60}")
    print(f"Testing input_layernorm for Layer {layer_idx}")
    print(f"{'='*60}")

    # Connect the device and create an xla mesh
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Load config and model
    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Load input
    if layer_idx == 0:
        print(f"\nLoading initial inputs from '{input_file}'...")
        captured_inputs = torch.load(input_file)
        hidden_states_cpu = captured_inputs["hidden_states"]
    else:
        print(f"\nLoading hidden states from '{input_file}'...")
        hidden_states_cpu = torch.load(input_file)

    print(f"Input shape: {hidden_states_cpu.shape}")

    # ========================================
    # Analyze Input Statistics
    # ========================================
    print(f"\n{'='*60}")
    print("Input Statistics")
    print(f"{'='*60}")
    print(f"Mean:     {hidden_states_cpu.mean().item():.6f}")
    print(f"Std:      {hidden_states_cpu.std().item():.6f}")
    print(f"Min:      {hidden_states_cpu.min().item():.6f}")
    print(f"Max:      {hidden_states_cpu.max().item():.6f}")
    print(f"Abs Max:  {hidden_states_cpu.abs().max().item():.6f}")

    # Per-token statistics
    token_means = hidden_states_cpu.mean(dim=-1)
    token_stds = hidden_states_cpu.std(dim=-1)
    print(
        f"\nPer-token mean range: [{token_means.min().item():.6f}, {token_means.max().item():.6f}]"
    )
    print(
        f"Per-token std range:  [{token_stds.min().item():.6f}, {token_stds.max().item():.6f}]"
    )

    # Get the layer
    layer = model.model.layers[layer_idx]
    input_ln = layer.input_layernorm

    print(f"\nRMSNorm parameters:")
    print(f"Weight shape: {input_ln.weight.shape}")
    print(f"Weight mean:  {input_ln.weight.mean().item():.6f}")
    print(f"Weight std:   {input_ln.weight.std().item():.6f}")
    print(f"Epsilon:      {input_ln.variance_epsilon}")

    # ========================================
    # CPU Reference with Intermediate Values
    # ========================================
    print(f"\n{'='*60}")
    print("CPU Reference Computation")
    print(f"{'='*60}")

    input_ln_cpu = input_ln.cpu()

    with torch.no_grad():
        # Replicate the forward pass to get intermediates
        input_dtype = hidden_states_cpu.dtype
        hidden_fp32 = hidden_states_cpu.to(torch.float32)

        # Compute variance
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        print(f"Variance statistics:")
        print(f"  Mean:     {variance.mean().item():.6e}")
        print(f"  Std:      {variance.std().item():.6e}")
        print(f"  Min:      {variance.min().item():.6e}")
        print(f"  Max:      {variance.max().item():.6e}")

        # Compute rsqrt
        rsqrt_var = torch.rsqrt(variance + input_ln_cpu.variance_epsilon)
        print(f"\nRsqrt(variance + eps) statistics:")
        print(f"  Mean:     {rsqrt_var.mean().item():.6f}")
        print(f"  Std:      {rsqrt_var.std().item():.6f}")
        print(f"  Min:      {rsqrt_var.min().item():.6f}")
        print(f"  Max:      {rsqrt_var.max().item():.6f}")

        # Normalized output
        normalized = hidden_fp32 * rsqrt_var
        print(f"\nNormalized (before weight) statistics:")
        print(f"  Mean:     {normalized.mean().item():.6f}")
        print(f"  Std:      {normalized.std().item():.6f}")
        print(f"  Min:      {normalized.min().item():.6f}")
        print(f"  Max:      {normalized.max().item():.6f}")

        # Final output
        output_cpu = (input_ln_cpu.weight * normalized).to(input_dtype)
        print(f"\nFinal output statistics:")
        print(f"  Mean:     {output_cpu.mean().item():.6f}")
        print(f"  Std:      {output_cpu.std().item():.6f}")
        print(f"  Min:      {output_cpu.min().item():.6f}")
        print(f"  Max:      {output_cpu.max().item():.6f}")

    # ========================================
    # TT Device Computation
    # ========================================
    print(f"\n{'='*60}")
    print("TT Device Computation")
    print(f"{'='*60}")

    input_ln_tt = layer.input_layernorm.to(device)
    xs.mark_sharding(input_ln_tt.weight, mesh, ("batch",))

    compiled_input_ln = torch.compile(input_ln_tt, backend="tt")

    hidden_states_tt = hidden_states_cpu.to(device)

    with torch.no_grad():
        output_tt = compiled_input_ln(hidden_states_tt)

    output_tt_cpu = output_tt.cpu()

    print(f"TT output statistics:")
    print(f"  Mean:     {output_tt_cpu.mean().item():.6f}")
    print(f"  Std:      {output_tt_cpu.std().item():.6f}")
    print(f"  Min:      {output_tt_cpu.min().item():.6f}")
    print(f"  Max:      {output_tt_cpu.max().item():.6f}")

    # ========================================
    # Comparison
    # ========================================
    print(f"\n{'='*60}")
    print("CPU vs TT Comparison")
    print(f"{'='*60}")

    pcc = compute_pcc(output_cpu, output_tt_cpu)
    print(f"PCC: {pcc:.6f}")

    # Element-wise error analysis
    abs_diff = (output_cpu - output_tt_cpu).abs()
    rel_diff = abs_diff / (output_cpu.abs() + 1e-8)

    print(f"\nAbsolute difference statistics:")
    print(f"  Mean:     {abs_diff.mean().item():.6e}")
    print(f"  Std:      {abs_diff.std().item():.6e}")
    print(f"  Min:      {abs_diff.min().item():.6e}")
    print(f"  Max:      {abs_diff.max().item():.6e}")
    print(f"  Median:   {abs_diff.median().item():.6e}")

    print(f"\nRelative difference statistics:")
    print(f"  Mean:     {rel_diff.mean().item():.6f}")
    print(f"  Std:      {rel_diff.std().item():.6f}")
    print(f"  Max:      {rel_diff.max().item():.6f}")

    # Find worst mismatches
    abs_diff_flat = abs_diff.flatten()
    worst_indices = abs_diff_flat.topk(5).indices
    print(f"\nTop 5 worst mismatches:")
    for i, idx in enumerate(worst_indices):
        cpu_val = output_cpu.flatten()[idx].item()
        tt_val = output_tt_cpu.flatten()[idx].item()
        diff = abs_diff_flat[idx].item()
        print(f"  {i+1}. CPU: {cpu_val:.6f}, TT: {tt_val:.6f}, Diff: {diff:.6e}")

    # ========================================
    # Save Debug Information
    # ========================================
    debug_info = {
        "layer_idx": layer_idx,
        "pcc": pcc,
        "input_stats": {
            "mean": hidden_states_cpu.mean().item(),
            "std": hidden_states_cpu.std().item(),
            "min": hidden_states_cpu.min().item(),
            "max": hidden_states_cpu.max().item(),
        },
        "variance_stats": {
            "mean": variance.mean().item(),
            "std": variance.std().item(),
            "min": variance.min().item(),
            "max": variance.max().item(),
        },
        "output_cpu_stats": {
            "mean": output_cpu.mean().item(),
            "std": output_cpu.std().item(),
            "min": output_cpu.min().item(),
            "max": output_cpu.max().item(),
        },
        "output_tt_stats": {
            "mean": output_tt_cpu.mean().item(),
            "std": output_tt_cpu.std().item(),
            "min": output_tt_cpu.min().item(),
            "max": output_tt_cpu.max().item(),
        },
        "error_stats": {
            "abs_diff_mean": abs_diff.mean().item(),
            "abs_diff_max": abs_diff.max().item(),
            "rel_diff_mean": rel_diff.mean().item(),
            "rel_diff_max": rel_diff.max().item(),
        },
    }

    debug_file = f"layernorm_debug/layer_{layer_idx}_input_ln_debug.pt"
    torch.save(debug_info, debug_file)
    print(f"\nDebug info saved to '{debug_file}'")

    # Save outputs
    torch.save(
        output_cpu, f"layernorm_debug/layer_{layer_idx}_input_ln_cpu_detailed.pt"
    )
    torch.save(
        output_tt_cpu, f"layernorm_debug/layer_{layer_idx}_input_ln_tt_detailed.pt"
    )
    torch.save(abs_diff, f"layernorm_debug/layer_{layer_idx}_input_ln_abs_diff.pt")

    print(f"{'='*60}\n")

    return debug_info


def test_post_attention_layernorm(layer_idx: int):
    """
    Deep debug test for post_attention_layernorm of a specific layer.
    Analyzes input statistics, intermediate computations, and output quality.

    Args:
        layer_idx: The layer index to test (0-based)
    """
    setup_spmd()

    # Create directory for saving outputs
    os.makedirs("layernorm_debug", exist_ok=True)

    # Check if the required input file exists (residual after attention)
    input_file = f"component_outputs/layer_{layer_idx}_attn_cpu.pt"

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please run test_layer_components({layer_idx}) first.")
        return

    # Also need the original hidden states for residual
    if layer_idx == 0:
        original_input_file = "first_layer_inputs.pt"
    else:
        original_input_file = f"cpu_hidden_states/layer_{layer_idx - 1}.pt"

    if not os.path.exists(original_input_file):
        print(f"Error: Original input file '{original_input_file}' not found!")
        return

    # Check if component outputs exist for verification
    cpu_output_file = f"component_outputs/layer_{layer_idx}_post_attn_ln_cpu.pt"
    tt_output_file = f"component_outputs/layer_{layer_idx}_post_attn_ln_tt.pt"
    if not os.path.exists(cpu_output_file) or not os.path.exists(tt_output_file):
        print(
            f"Warning: Component output files not found. Run test_layer_components({layer_idx}) first for comparison."
        )

    print(f"\n{'='*60}")
    print(f"Testing post_attention_layernorm for Layer {layer_idx}")
    print(f"{'='*60}")

    # Connect the device and create an xla mesh
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Load config and model
    quantization_config = Mxfp4Config(dequantize=True)
    config = AutoConfig.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        "openai/gpt-oss-120b",
        config=config,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model.eval()

    # Load inputs
    print(f"\nLoading attention output from '{input_file}'...")
    attn_output_cpu = torch.load(input_file)

    if layer_idx == 0:
        captured_inputs = torch.load(original_input_file)
        original_hidden_states = captured_inputs["hidden_states"]
    else:
        original_hidden_states = torch.load(original_input_file)

    # Compute residual (input to post_attention_layernorm)
    residual_cpu = original_hidden_states + attn_output_cpu

    print(f"Residual shape: {residual_cpu.shape}")

    # ========================================
    # Analyze Input Statistics
    # ========================================
    print(f"\n{'='*60}")
    print("Input (Residual) Statistics")
    print(f"{'='*60}")
    print(f"Mean:     {residual_cpu.mean().item():.6f}")
    print(f"Std:      {residual_cpu.std().item():.6f}")
    print(f"Min:      {residual_cpu.min().item():.6f}")
    print(f"Max:      {residual_cpu.max().item():.6f}")
    print(f"Abs Max:  {residual_cpu.abs().max().item():.6f}")

    # Per-token statistics
    token_means = residual_cpu.mean(dim=-1)
    token_stds = residual_cpu.std(dim=-1)
    print(
        f"\nPer-token mean range: [{token_means.min().item():.6f}, {token_means.max().item():.6f}]"
    )
    print(
        f"Per-token std range:  [{token_stds.min().item():.6f}, {token_stds.max().item():.6f}]"
    )

    # Get the layer
    layer = model.model.layers[layer_idx]
    post_attn_ln = layer.post_attention_layernorm

    print(f"\nRMSNorm parameters:")
    print(f"Weight shape: {post_attn_ln.weight.shape}")
    print(f"Weight mean:  {post_attn_ln.weight.mean().item():.6f}")
    print(f"Weight std:   {post_attn_ln.weight.std().item():.6f}")
    print(f"Epsilon:      {post_attn_ln.variance_epsilon}")

    # ========================================
    # CPU Reference with Intermediate Values
    # ========================================
    print(f"\n{'='*60}")
    print("CPU Reference Computation")
    print(f"{'='*60}")

    post_attn_ln_cpu = post_attn_ln.cpu()

    with torch.no_grad():
        # Replicate the forward pass to get intermediates
        input_dtype = residual_cpu.dtype
        hidden_fp32 = residual_cpu.to(torch.float32)

        # Compute variance
        variance = hidden_fp32.pow(2).mean(-1, keepdim=True)
        print(f"Variance statistics:")
        print(f"  Mean:     {variance.mean().item():.6e}")
        print(f"  Std:      {variance.std().item():.6e}")
        print(f"  Min:      {variance.min().item():.6e}")
        print(f"  Max:      {variance.max().item():.6e}")

        # Compute rsqrt
        rsqrt_var = torch.rsqrt(variance + post_attn_ln_cpu.variance_epsilon)
        print(f"\nRsqrt(variance + eps) statistics:")
        print(f"  Mean:     {rsqrt_var.mean().item():.6f}")
        print(f"  Std:      {rsqrt_var.std().item():.6f}")
        print(f"  Min:      {rsqrt_var.min().item():.6f}")
        print(f"  Max:      {rsqrt_var.max().item():.6f}")

        # Normalized output
        normalized = hidden_fp32 * rsqrt_var
        print(f"\nNormalized (before weight) statistics:")
        print(f"  Mean:     {normalized.mean().item():.6f}")
        print(f"  Std:      {normalized.std().item():.6f}")
        print(f"  Min:      {normalized.min().item():.6f}")
        print(f"  Max:      {normalized.max().item():.6f}")

        # Final output
        output_cpu = (post_attn_ln_cpu.weight * normalized).to(input_dtype)
        print(f"\nFinal output statistics:")
        print(f"  Mean:     {output_cpu.mean().item():.6f}")
        print(f"  Std:      {output_cpu.std().item():.6f}")
        print(f"  Min:      {output_cpu.min().item():.6f}")
        print(f"  Max:      {output_cpu.max().item():.6f}")

    # ========================================
    # TT Device Computation
    # ========================================
    print(f"\n{'='*60}")
    print("TT Device Computation")
    print(f"{'='*60}")

    post_attn_ln_tt = layer.post_attention_layernorm.to(device)
    xs.mark_sharding(post_attn_ln_tt.weight, mesh, ("batch",))
    compiled_post_attn_ln = torch.compile(post_attn_ln_tt, backend="tt")

    residual_tt = residual_cpu.to(device)

    with torch.no_grad():
        output_tt = compiled_post_attn_ln(residual_tt)

    output_tt_cpu = output_tt.cpu()

    print(f"TT output statistics:")
    print(f"  Mean:     {output_tt_cpu.mean().item():.6f}")
    print(f"  Std:      {output_tt_cpu.std().item():.6f}")
    print(f"  Min:      {output_tt_cpu.min().item():.6f}")
    print(f"  Max:      {output_tt_cpu.max().item():.6f}")

    # ========================================
    # Comparison
    # ========================================
    print(f"\n{'='*60}")
    print("CPU vs TT Comparison")
    print(f"{'='*60}")

    pcc = compute_pcc(output_cpu, output_tt_cpu)
    print(f"PCC: {pcc:.6f}")

    # Element-wise error analysis
    abs_diff = (output_cpu - output_tt_cpu).abs()
    rel_diff = abs_diff / (output_cpu.abs() + 1e-8)

    print(f"\nAbsolute difference statistics:")
    print(f"  Mean:     {abs_diff.mean().item():.6e}")
    print(f"  Std:      {abs_diff.std().item():.6e}")
    print(f"  Min:      {abs_diff.min().item():.6e}")
    print(f"  Max:      {abs_diff.max().item():.6e}")
    print(f"  Median:   {abs_diff.median().item():.6e}")

    print(f"\nRelative difference statistics:")
    print(f"  Mean:     {rel_diff.mean().item():.6f}")
    print(f"  Std:      {rel_diff.std().item():.6f}")
    print(f"  Max:      {rel_diff.max().item():.6f}")

    # Find worst mismatches
    abs_diff_flat = abs_diff.flatten()
    worst_indices = abs_diff_flat.topk(5).indices
    print(f"\nTop 5 worst mismatches:")
    for i, idx in enumerate(worst_indices):
        cpu_val = output_cpu.flatten()[idx].item()
        tt_val = output_tt_cpu.flatten()[idx].item()
        diff = abs_diff_flat[idx].item()
        print(f"  {i+1}. CPU: {cpu_val:.6f}, TT: {tt_val:.6f}, Diff: {diff:.6e}")

    # ========================================
    # Save Debug Information
    # ========================================
    debug_info = {
        "layer_idx": layer_idx,
        "pcc": pcc,
        "input_stats": {
            "mean": residual_cpu.mean().item(),
            "std": residual_cpu.std().item(),
            "min": residual_cpu.min().item(),
            "max": residual_cpu.max().item(),
        },
        "variance_stats": {
            "mean": variance.mean().item(),
            "std": variance.std().item(),
            "min": variance.min().item(),
            "max": variance.max().item(),
        },
        "output_cpu_stats": {
            "mean": output_cpu.mean().item(),
            "std": output_cpu.std().item(),
            "min": output_cpu.min().item(),
            "max": output_cpu.max().item(),
        },
        "output_tt_stats": {
            "mean": output_tt_cpu.mean().item(),
            "std": output_tt_cpu.std().item(),
            "min": output_tt_cpu.min().item(),
            "max": output_tt_cpu.max().item(),
        },
        "error_stats": {
            "abs_diff_mean": abs_diff.mean().item(),
            "abs_diff_max": abs_diff.max().item(),
            "rel_diff_mean": rel_diff.mean().item(),
            "rel_diff_max": rel_diff.max().item(),
        },
    }

    debug_file = f"layernorm_debug/layer_{layer_idx}_post_attn_ln_debug.pt"
    torch.save(debug_info, debug_file)
    print(f"\nDebug info saved to '{debug_file}'")

    # Save outputs
    torch.save(
        output_cpu, f"layernorm_debug/layer_{layer_idx}_post_attn_ln_cpu_detailed.pt"
    )
    torch.save(
        output_tt_cpu, f"layernorm_debug/layer_{layer_idx}_post_attn_ln_tt_detailed.pt"
    )
    torch.save(abs_diff, f"layernorm_debug/layer_{layer_idx}_post_attn_ln_abs_diff.pt")

    print(f"{'='*60}\n")

    return debug_info


if __name__ == "__main__":
    xr.set_device_type("TT")
    test_input_layernorm(7)
    # test_post_attention_layernorm(7)
    # test_layer_components_from_cpu_output(1)
    # test_layer_components(0)
    # test_specific_layer(6)
    # test_single_layer_multiple_runs()
