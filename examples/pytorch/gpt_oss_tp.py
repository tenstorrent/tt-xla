# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# This demo runs a single forward pass on the Llama 3.1 8B parameter model eagerly
# using Torch-XLA's SPMD mode.
import os
import sys
import torch
import torch_xla
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs
from torch_xla.distributed.spmd import Mesh
from torch_xla.experimental import plugins
import numpy as np
from transformers import AutoTokenizer, GptOssForCausalLM, GptOssModel, GptOssConfig

def setup_xla_environment():
    """Setup XLA environment for tensor parallelism."""
    # Basic XLA configuration
    os.environ["PJRT_DEVICE"] = "TT"
    os.environ["XLA_STABLEHLO_COMPILE"] = "1"
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    class TTPjrtPlugin(plugins.DevicePlugin):
        def library_path(self):
            return os.path.join(
                os.path.dirname(__file__), "../../build/src/tt/pjrt_plugin_tt.so"
            )

    plugins.register_plugin("TT", TTPjrtPlugin())

    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def apply_tensor_parallel_sharding_causal(causal_model: GptOssForCausalLM, mesh: Mesh) -> None:
    """
    Apply tensor parallel sharding to the causal Llama model (specifically to the MLP,
    self-attention, and LM heads).
    """
    # Move model to XLA device first
    causal_model = causal_model.to(torch_xla.device())

    # Shard the base model first
    apply_tensor_parallel_sharding_base(causal_model.model, mesh)

    # Now shard the LM head
    # lm_head: [vocab_size, hidden_size] -> shard dim 0
    xs.mark_sharding(causal_model.lm_head.weight, mesh, ("model", None))
    print("Tensor parallel sharding applied successfully!")


def apply_tensor_parallel_sharding_base(base_model: GptOssModel, mesh: Mesh) -> None:
    """
    Apply tensor parallel sharding to the base Llama model.
    """
    # Apply sharding to each transformer layer
    for layer in base_model.layers:
        # ========================================
        # MLP (Feed-Forward) Layer Sharding - shard the intermediate_size across devices
        # ========================================

        # EP try

        # Replicate all router matrices
        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, None)) # [32, 2880]
        xs.mark_sharding(layer.mlp.router.bias, mesh, (None,)) # [32]

        # Shard all expert matrices on the experts dimension (dim 0)
        # [32, 2880, 5760]
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", None, None))
        # [32, 5760]
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        # [32, 2880, 2880]
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, None))
        # [32, 2880]
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", None))

        
        # TP try
        # xs.mark_sharding(layer.mlp.router.weight, mesh, (None, None))
        # xs.mark_sharding(layer.mlp.router.bias, mesh, (None,))
        # xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, (None, None, None))
        # xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, (None, None))
        # xs.mark_sharding(layer.mlp.experts.down_proj, mesh, (None, None, "model"))
        # xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, (None, "model"))

        # ========================================
        # Self-Attention Layer Sharding - shard the heads across all devices
        # ========================================

        # q_proj: [num_heads * head_dim, hidden_size] -> shard dim 0
        # [4096, 2880]
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))

        # k_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        # [512, 2880]
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))

        # v_proj: [num_kv_heads * head_dim, hidden_size] -> shard dim 0
        # [512, 2880]
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))

        # o_proj: [hidden_size, num_heads * head_dim] -> shard dim 1
        # [2880, 4096]
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

        # sinks: [num_heads] -> shard dim 0
        # [64]
        xs.mark_sharding(layer.self_attn.sinks, mesh, ("model",))

def prepare_inputs(mesh: Mesh, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Prepare input tensors by replicating them across the device mesh.
    """
    print(f"Preparing inputs for TP: batch_size={input_ids.shape[0]}, seq_length={input_ids.shape[1]}")

    # Move to XLA device
    input_ids = input_ids.to(torch_xla.device())

    # Replicate inputs to all devices
    xs.mark_sharding(input_ids, mesh, (None, None))

    return input_ids


def decode_output(logits: torch.Tensor, tokenizer: AutoTokenizer):
    """
    Helper function to decode the output logits to text
    """
    next_token_logits = logits[:, -1, :]
    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
    decoded_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)

    return decoded_text, next_token_id


def run_gpt_oss_tp():
    """
    Run a single forward pass using tensor parallelism.
    """
    model_name = "openai/gpt-oss-20b"
    prompt = "What is the name of the largest planet in our solar system?"

    # Setup environment
    setup_xla_environment()
    mesh = create_device_mesh()

    # Load model and configuration
    print("Loading model...")
    config = GptOssConfig.from_pretrained(model_name)
    # Delete quantization config since mxfp4 quantization is not supported
    delattr(config, "quantization_config")
    config.num_hidden_layers = 1
    # config.num_local_experts = 32
    model = GptOssForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, config=config)
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)

    print("Running Tensor Parallel Inference")

    # Apply sharding to model
    apply_tensor_parallel_sharding_causal(model, mesh)

    # Prepare inputs for tensor parallel execution
    input_ids_tp = prepare_inputs(mesh, input_ids)

    # Run tensor parallel inference
    with torch.no_grad():
        outputs_tp = model(input_ids=input_ids_tp)
        torch_xla.sync()  # Ensure all computations are done
        # Move the outputs to CPU
        tp_logits = outputs_tp.logits.to("cpu")
    tp_text, tp_next_token = decode_output(tp_logits, tokenizer)

    print("\n=== Results ===")
    print("Input Prompt: ", prompt)
    print("Output text: ", tp_text)
    print("Output token ID: ", tp_next_token.item())


def main():
    print("Torch-XLA SPMD Tensor Parallelism for GPT-OSS 20B Model")
    print("=" * 50)

    try:
        run_gpt_oss_tp()
    except Exception as e:
        print(f"Error during execution: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())