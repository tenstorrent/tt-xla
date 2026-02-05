# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
from typing import Any, List, Optional

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

DEFAULT_PROMPTS = ["Explain quantum mechanics."]


# --------------------------------
# GPT-OSS 20B Generation Loop Example
# --------------------------------
def gpt_oss_20b(interactive: bool = False):

    # Set up config variables.
    max_cache_len: int = 256
    model_name: str = "openai/gpt-oss-20b"

    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    # Instantiate model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)

    while True:
        if interactive:
            user_prompt = input("Enter your prompt or quit() to exit: ")
            batch_size: int = 1
            if user_prompt.lower() == "quit()":
                break
            user_prompt = [user_prompt]
        else:
            user_prompt = DEFAULT_PROMPTS
            batch_size: int = len(user_prompt)

        # Construct inputs, including static cache
        input_args, formatted_prompts = construct_inputs(
            user_prompt, tokenizer, model.config, batch_size, max_cache_len
        )

        # Limit maximum generation count to fit within preallocated static cache
        max_tokens_to_generate: int = max_cache_len - input_args["input_ids"].shape[1]

        # Transfer model and inputs to device
        model, input_args = transfer_to_device(model, input_args, device)

        # Mark sharding on inputs and model internals
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
            max_tokens_to_generate,
            formatted_prompts,
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

    if num_devices == 32:  # Galaxy
        mesh_shape = (8, 4)
    elif num_devices == 8:  # llmbox
        mesh_shape = (2, 4)
    else:
        raise RuntimeError(f"Gpt-oss is only supported on llmbox and galaxy")

    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


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
    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = model.eval()

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def construct_inputs(
    input_prompt: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
) -> tuple[dict, List[str]]:
    """
    Construct inputs including static cache.

    Args:
        input_prompt: Input text prompt(s) - can be a single string or list of strings
        tokenizer: Tokenizer instance
        model_config: Model configuration
        batch_size: Batch size
        max_cache_len: Maximum cache length

    Returns:
        Tuple of (input_args dictionary, formatted_prompts list)
    """

    # Apply chat template to format prompts
    formatted_prompts = []
    for prompt in input_prompt:
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    prompt_lengths = [
        len(tokenizer.encode(p, add_special_tokens=False)) for p in formatted_prompts
    ]
    max_length = max(prompt_lengths)

    inputs = tokenizer(
        formatted_prompts,
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # Disable sliding window cache to avoid torch.compile recompilations.
    cache_config = disable_sliding_window_attention(model_config)

    # Static cache should be initialized on CPU and separately transferred to device
    # due to a trace/fusion issue. See https://github.com/tenstorrent/tt-xla/issues/1645
    static_cache: StaticCache = StaticCache(
        config=cache_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    cache_position: torch.Tensor = torch.arange(0, inputs.input_ids.shape[1])

    # Attention mask is needed to ignore padding tokens in left-padded batches. The mask should match max_cache_len
    # to prevent recompilation or implicit padding by transformers, which can cause degenerate output.
    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    #   Debug prints
    print("\n=== DEBUG: construct_inputs ===")
    print(f"Original prompts: {input_prompt}")
    print(f"Formatted prompts (with chat template): {formatted_prompts}")
    print(f"Input IDs shape: {inputs.input_ids.shape}")
    print(f"Input IDs: {inputs.input_ids}")
    print(f"Input attention mask shape: {inputs.attention_mask.shape}")
    print(f"Full attention mask shape (pre-allocated): {full_attention_mask.shape}")
    print(f"Full attention mask: {full_attention_mask}")
    print(f"Cache position shape: {cache_position.shape}")
    print(f"Cache position: {cache_position}")
    print(f"Actual sequence length (non-padding): {inputs.attention_mask.sum().item()}")
    print("=" * 50)

    return input_args, formatted_prompts


def disable_sliding_window_attention(
    config: PretrainedConfig,
) -> PretrainedConfig:
    """
    Override all layer types to 'full_attention' to disable sliding window cache.

    GPT-OSS-20B originally has 24 layers alternating between "sliding_attention" and
    "full_attention". When StaticCache is initialized with sliding attention layers,
    it creates StaticSlidingWindowLayer instances that use conditional logic and
    Python Int (which torch.compile treats as constants). This triggers expensive
    recompilation for every token generated.

    This function offers a hacky workaround. Because this program stops token
    generation once the cache is full, StaticSlidingWindowLayer would be essentially
    functionally equivalent to StaticLayer. Hence by forcing all layers to be
    "full_attention" before passing the config to create the StaticCache, we can
    ensure the cache only has simple StaticLayer and avoid recompilation.

    Args:
        config: Model configuration with layer_types attribute

    Returns:
        Config modified to have all layers set to "full_attention"
    """
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    return config


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

    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, "batch"))
    xs.mark_sharding(model.model.norm.weight, mesh, ("batch",))
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", "batch"))

    for layer in model.model.layers:
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.q_proj.bias, mesh, ("model",))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.k_proj.bias, mesh, ("model",))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", "batch"))
        xs.mark_sharding(layer.self_attn.v_proj.bias, mesh, ("model",))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, ("batch", "model"))
        xs.mark_sharding(layer.self_attn.o_proj.bias, mesh, ("batch",))

        xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

        xs.mark_sharding(layer.mlp.router.weight, mesh, (None, "batch"))

        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, ("model", "batch", None))
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, ("model", None, "batch"))
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, ("model", "batch"))

        xs.mark_sharding(layer.input_layernorm.weight, mesh, ("batch",))
        xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, ("batch",))


def run_generate(
    compiled_model: torch.nn.Module,
    input_args: dict,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    mesh: Mesh = None,
    max_tokens_to_generate: int = 128,
    formatted_prompts: List[str] = [""],
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
        max_tokens_to_generate: Maximum number of tokens to generate
        formatted_prompts: Formatted prompts with chat template applied
        is_interactive: Whether running in interactive mode
    """
    num_users = input_args["input_ids"].shape[0]
    output_tokens: List[List[str]] = [[] for _ in range(num_users)]

    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            if step == 0:
                print("RUNNING PREFILL")
                if is_interactive:
                    print("=" * 80)
                    print("PROMPT:")
                    print(formatted_prompts[0])
                    print("-" * 80)
                    print("GENERATED:", end="", flush=True)

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
            print(f"=" * 80)
            print(f"Result for user {i}:")
            print(f"-" * 80)
            print("PROMPT:")
            print(formatted_prompts[i])
            print(f"-" * 80)
            print("GENERATED:")
            print("".join(output_tokens[i]))
            print(f"=" * 80)
            print()


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
