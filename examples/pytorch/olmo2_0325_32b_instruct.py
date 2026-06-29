# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel text-generation example for allenai/OLMo-2-0325-32B-Instruct.

This drives the model end-to-end through the tt-forge-models loader API:
the loader supplies the weights, tokenizer, config, the (1 x N) device mesh,
and the Megatron column->row tensor-parallel shard plan. We then run a real
greedy decode loop (prefill + per-token decode) over a StaticCache, sharding
the model across all visible chips so the 32B weights fit.

OLMo-2 uses full (non-sliding) attention, so no sliding-window cache overrides
are needed (unlike the OLMo-3 example).
"""

import os
from typing import List

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig

from third_party.tt_forge_models.olmo2.causal_lm.pytorch import (
    ModelLoader,
    ModelVariant,
)

MAX_LENGTH = 256
BATCH_SIZE = 1
MAX_TOKENS_TO_GENERATE = 20


def setup_spmd():
    """Initializes SPMD mode in torch_xla (same as examples/pytorch/llama.py)."""
    print("Setting up XLA environment...")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()
    print("XLA environment configured.")


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module,
    input_args: dict,
    loader: ModelLoader,
    mesh: Mesh,
):
    """Apply the loader's tensor-parallel shard plan to weights and KV cache.

    Model weights use the loader's public Megatron column->row plan. KV cache
    tensors are sharded on the head dimension (num_key_value_heads is divisible
    by the device count). Tensors left unmarked (input_ids, cache_position, ...)
    stay replicated.
    """
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    for layer in input_args["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))


# --------------------------------
# OLMo-2 0325 32B Instruct generation loop example
# --------------------------------
def olmo2_0325_32b_instruct():
    loader = ModelLoader(ModelVariant.Olmo_2_0325_32B_Instruct)

    num_devices = xr.global_runtime_device_count()
    is_spmd = num_devices > 1
    if is_spmd:
        setup_spmd()

    device: torch.device = torch_xla.device()

    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    tokenizer = loader.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    mesh: Mesh | None = None
    if is_spmd:
        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, mesh_names)
        print(f"Created device mesh: {mesh_shape} with {num_devices} devices")

    input_args, formatted_prompts = _prepare_inputs(
        [loader.sample_text], tokenizer, model.config, BATCH_SIZE, MAX_LENGTH
    )

    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)
        if hasattr(layer, "device"):
            layer.device = device

    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["position_ids"] = input_args["position_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)

    if is_spmd and mesh is not None:
        mark_sharding_on_inputs_and_model(model, input_args, loader, mesh)

    # Match the bringup baseline: bf8 weights keep the 32B compile within budget.
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")

    output_tokens: List[str] = []
    with torch.no_grad():
        for step in range(MAX_TOKENS_TO_GENERATE):
            output = compiled_model(**input_args)
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True
            )
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            output_tokens.append(tokenizer.decode(next_token_id[0]))

            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # newline after generation completes
                break

            # Update inputs for next iteration.
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

            host_cache_pos = input_args["cache_position"].to("cpu")
            host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
            input_args["cache_position"] = host_cache_pos.to(device)
            # Keep position_ids in sync with cache_position (see _prepare_inputs).
            input_args["position_ids"] = host_cache_pos.unsqueeze(0).to(device)

    generated_text = "".join(output_tokens)
    post_process_output(formatted_prompts[0], generated_text)
    return generated_text


def post_process_output(prompt: str, generated_text: str):
    """Print the human-readable generation result."""
    print("=" * 80)
    print("PROMPT:")
    print(prompt)
    print("-" * 80)
    print("GENERATED:")
    print(generated_text)
    print("=" * 80)


def _prepare_inputs(
    input_prompt: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
) -> tuple[dict, List[str]]:

    formatted_prompts = []
    for prompt in input_prompt:
        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
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
    seq_len = inputs.input_ids.shape[1]

    static_cache = StaticCache(
        config=model_config,
        max_cache_len=max_cache_len,
    )

    text_config = model_config.get_text_config(decoder=True)
    head_dim = getattr(text_config, "head_dim", None) or (
        text_config.hidden_size // text_config.num_attention_heads
    )
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=text_config.num_key_value_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    # Attention mask must span max_cache_len to keep a single static shape and
    # avoid recompilation / degenerate output from implicit padding.
    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    cache_position = torch.arange(0, seq_len)
    # Pass position_ids explicitly so the forward never enters the
    # "position_ids is None" branch (which bakes a per-step Python int into the
    # graph and forces a recompile on every decode step).
    position_ids = cache_position.unsqueeze(0)

    input_args = {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "position_ids": position_ids,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }
    return input_args, formatted_prompts


def test_olmo2_0325_32b_instruct():
    """Smoke test: the TP generation loop produces non-empty, finite output."""
    xr.set_device_type("TT")

    generated_text = olmo2_0325_32b_instruct()

    assert isinstance(generated_text, str)
    assert len(generated_text.strip()) > 0, "Generation produced no text"


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    olmo2_0325_32b_instruct()
