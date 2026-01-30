# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import List

import numpy as np
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
import transformers
from loguru import logger
from torch_xla.distributed.spmd import Mesh
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

DEFAULT_CHAT_MESSAGES = [
    [
        {
            "role": "system",
            "content": "You are ChatGPT, a large language model trained by OpenAI.",
        },
        {"role": "user", "content": "Explain quantum mechanics clearly and concisely."},
    ],
]


# --------------------------------
# GPT-OSS 20B Generation Loop Example
# --------------------------------
def gpt_oss_20b(interactive: bool = False):

    # TODO: no need to check version?
    # Check transformers version
    # check_transformers_version()

    # Set up config variables.
    # Increased cache length to accommodate chat template overhead (~100 tokens)
    # and still allow reasonable generation length
    max_cache_len: int = 200  # Increased from 128
    model_name: str = "openai/gpt-oss-20b"

    # TODO: always need spmd?
    # Determine if SPMD mode should be enabled, if more than 1 device is available.
    # SPMD must be turned off for llama generate on 1x1 mesh - See https://github.com/tenstorrent/tt-xla/issues/1639
    num_devices = xr.global_runtime_device_count()
    is_spmd: bool = num_devices > 1
    if is_spmd:
        setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    # Debug: Check what attention implementation is actually being used
    print(f"Model config._attn_implementation: {model.config._attn_implementation}")

    while True:
        if interactive:
            user_input = input("Enter your prompt or quit() to exit: ")
            batch_size: int = 1
            if user_input.lower() == "quit()":
                break
            # Format as chat messages
            user_messages = [
                [
                    {
                        "role": "system",
                        "content": "You are ChatGPT, a large language model trained by OpenAI.",
                    },
                    {"role": "user", "content": user_input},
                ]
            ]
        else:
            batch_size: int = 1  # Reduced from 32 to avoid OOM
            user_messages = DEFAULT_CHAT_MESSAGES[:batch_size]

        # Construct inputs, including static cache
        input_args = construct_inputs(
            user_messages, tokenizer, model.config, batch_size, max_cache_len
        )

        # Limit maximum generation count to fit within preallocated static cache
        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        # Transfer model and inputs to device
        model, input_args = transfer_to_device(model, input_args, device)

        # Mark sharding on inputs and model internals if SPMD is enabled
        if is_spmd:
            mark_sharding_on_inputs_and_model(model, input_args, mesh)

        # Compile model
        compiled_model = torch.compile(model, backend="tt")

        # Run generation loop until EOS token generated or max tokens reached
        run_generate(
            compiled_model,
            input_args,
            tokenizer,
            device,
            mesh,
            is_spmd,
            max_tokens_to_generate,
            user_messages,
            interactive,
        )

        if not interactive:
            break


def setup_spmd():
    """
    Initializes SPMD mode in torch_xla.
    """

    print("Setting up XLA environment...")

    # Converts the StableHLO emitted by torch-xla to the Shardy dialect
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"

    # Initialize SPMD
    xr.use_spmd()
    print("XLA environment configured.")


def create_device_mesh() -> Mesh:
    """
    Create device mesh for tensor parallelism.

    Returns:
        Mesh object for SPMD operations
    """
    num_devices = xr.global_runtime_device_count()
    mesh_shape = (1, num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def load_config(model_name: str) -> PretrainedConfig:
    """Load and return the configuration for the gpt-oss model with this instance's variant.

    Returns:
        The configuration object for the gpt-oss model.
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    config.quantization_config["quant_method"] = "none"
    # TODO: unnecessary?
    config.use_cache = True
    return config


def setup_model_and_tokenizer(
    model_name: str,
) -> tuple[torch.nn.Module, PreTrainedTokenizer]:
    """
    Instantiate model and tokenizer.

    Args:
        model_name: HuggingFace model name

    Returns:
        Tuple of (model, tokenizer)
    """
    config = load_config(model_name)
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        # attn_implementation="eager",  # Removed - use default like Llama
    )
    model = model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    chat_messages: List[List[dict]],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
) -> dict:
    """
    Construct inputs including static cache.

    Args:
        chat_messages: List of chat message lists, where each chat message list contains dicts with 'role' and 'content' keys
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length

    Returns:
        Dictionary containing input_ids, past_key_values, cache_position, and use_cache
    """
    # Apply chat template to each message list
    formatted_prompts = []
    for messages in chat_messages:
        # Apply chat template - this formats messages in harmony response format
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,  # Add prompt for model to generate response
        )
        formatted_prompts.append(formatted_prompt)

    # Tokenize the formatted prompts
    # Note: Chat templates can produce 80-100+ tokens due to harmony format overhead
    # We need enough space for: system message + user message + template markers
    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        max_length=100,  # Increased from 32 to accommodate chat template
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # Static cache should be initialized on CPU and separately transferred to device
    # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache: StaticCache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_key_value_heads = model_config.num_key_value_heads
    # head_dim = model_config.hidden_size // model_config.num_attention_heads
    head_dim = model_config.head_dim
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )
    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    # Create 4D attention mask to bypass get_mask_sizes() and prevent recompilations
    # 4D mask triggers early exit in masking_utils.py:709, avoiding cumulative_length access
    # Shape: (batch_size, num_heads=1, query_length, key_value_length)
    prompt_len = inputs.input_ids.shape[1]

    # Initialize 4D mask: (batch, 1, query_len, kv_len)
    # Use float dtype with 0.0 = can attend, -inf = masked out
    full_attention_mask = torch.zeros(
        (batch_size, 1, prompt_len, max_cache_len), dtype=torch.float32
    )

    # For each batch element, mask out padding positions
    for i in range(batch_size):
        # Count valid (non-padding) tokens in this sequence
        # inputs.attention_mask is 1 for valid tokens, 0 for padding
        valid_token_count = inputs.attention_mask[i].sum().item()

        # Allow attending to valid token positions
        # valid tokens are at the END for left-padded sequences
        valid_start_pos = prompt_len - valid_token_count

        # Set attention to 0.0 (can attend) for valid positions
        # Set to -inf (cannot attend) for padding positions
        full_attention_mask[i, 0, :, valid_start_pos:prompt_len] = 0.0
        full_attention_mask[i, 0, :, :valid_start_pos] = float("-inf")  # Mask padding
        # TODO: should be 0 instead of -inf?
        full_attention_mask[i, 0, :, prompt_len:] = float(
            "-inf"
        )  # Mask future cache positions

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        # TODO: no need?
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    #   Debug prints
    print("\n=== DEBUG: construct_inputs ===")
    print(f"Chat messages: {chat_messages}")
    print(f"Formatted prompts: {formatted_prompts}")
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs: {inputs.input_ids}")
    print(f"Input attention mask shape: {inputs.attention_mask.shape}")
    print(f"Full attention mask shape (pre-allocated): {full_attention_mask.shape}")
    print(f"Full attention mask: {full_attention_mask}")
    print(f"Cache position shape: {cache_position.shape}")
    print(f"Cache position: {cache_position}")
    print(f"Actual sequence length (non-padding): {inputs.attention_mask.sum().item()}")
    print("=" * 50)

    return input_args


def transfer_to_device(
    model: torch.nn.Module, input_args: dict, device: torch.device
) -> tuple[torch.nn.Module, dict]:
    """
    Transfer model and inputs to device.

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        device: Target device

    Returns:
        Tuple of (model, input_args) on device
    """
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)

    model = model.to(device)

    return model, input_args


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module, input_args: dict, mesh: Mesh
):
    """
    Mark sharding on inputs and model internals.
    If mark_sharding is not called on a tensor, it is fully replicated across all devices.
        i.e. on cache_positions, input_ids

    Args:
        model: Model instance
        input_args: Input arguments dictionary
        mesh: Device mesh for SPMD operations
    """

    for layer in input_args["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    # Shard model internals
    for layer in model.model.layers:
        # xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        # xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        # xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, None))
        xs.mark_sharding(
            layer.mlp.experts.gate_up_proj,
            mesh,
            (
                "model",
                None,
                None,
            ),
        )
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        xs.mark_sharding(
            layer.mlp.experts.down_proj,
            mesh,
            (
                "model",
                None,
                None,
            ),
        )
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", None))


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh = None,
    is_spmd: bool = True,
    max_tokens_to_generate: int = 128,
    chat_messages: List[List[dict]] = [[]],
    is_interactive: bool = False,
):
    """
    Run the generation loop.

    Args:
        compiled_model: Compiled model instance
        input_args: Input arguments dictionary
        tokenizer: Tokenizer instance
        device: Device
        mesh: Device mesh for SPMD operations (optional)
        is_spmd: Whether SPMD mode is enabled
        max_tokens_to_generate: Maximum number of tokens to generate
        chat_messages: List of chat message lists for each user in the batch
        is_interactive: Whether in interactive mode
    """
    num_users = input_args["input_ids"].shape[0]
    output_tokens: List[List[str]] = [[] for _ in range(num_users)]
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("RUNNING PREFILL")
                if is_interactive:
                    # Extract user message content from the first chat in batch
                    user_content = next(
                        (
                            msg["content"]
                            for msg in chat_messages[0]
                            if msg["role"] == "user"
                        ),
                        "",
                    )
                    print(f"User: {user_content}")
                    print(f"Assistant: ", end="", flush=True)

            # Run forward pass
            output: CausalLMOutputWithPast = compiled_model(**input_args)
            output_logits: torch.Tensor = output.logits.to("cpu")
            next_token_id = output_logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[i]) for i in range(num_users)]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])
                if is_interactive:
                    print(output_text[i], end="", flush=True)

            # Check for EOS token and early exit
            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # Add newline after generation completes
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)

    print()
    if not is_interactive:
        for i in range(num_users):
            # Extract user message content
            user_content = next(
                (msg["content"] for msg in chat_messages[i] if msg["role"] == "user"),
                "",
            )
            print(f"User {i}: {user_content}")
            print(f"Assistant: {''.join(output_tokens[i])}")
            print()


# def check_transformers_version():
#     """
#     Check that transformers version is <= 4.52.4.
#     Raises RuntimeError if version is incompatible.

#     This is because transformers SDPA implementation changed in later versions,
#     which causes dynamo trace to fail.

#     See https://github.com/tenstorrent/tt-xla/issues/1020
#     """
#     import packaging.version

#     current_version = packaging.version.parse(transformers.__version__)
#     max_version = packaging.version.parse("4.57.1")

#     if current_version > max_version:
#         raise RuntimeError(
#             f"Transformers version {transformers.__version__} is not supported. "
#             f"Please use version <= 4.57.1"
#         )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT-OSS 20B generation example")
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Enable interactive mode for entering custom prompts",
    )
    args = parser.parse_args()

    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    gpt_oss_20b(interactive=args.interactive)
