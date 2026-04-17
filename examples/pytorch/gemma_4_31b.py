# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import copy
import os
from typing import List

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache
from transformers.configuration_utils import PretrainedConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from tt_torch.transformers_overrides import (
    override_cache_sliding_window_layers,
    override_gemma4_sliding_window_causal_mask,
)

MODEL_NAME = "google/gemma-4-31B-it"
DEFAULT_PROMPT = "What is your favorite city?"
MAX_LENGTH = 256
BATCH_SIZE = 1


# --------------------------------
# Gemma 4 E4B Generation Loop Example
# --------------------------------
def gemma_4_31b():
    # Set up config variables.
    model_name = "google/gemma-4-31B-it"

    setup_spmd()

    # Connect the device and create an xla mesh.
    device: torch.device = torch_xla.device()
    mesh: Mesh = create_device_mesh()

    print("Loading model and tokenizer...", flush=True)
    model, tokenizer = _load_model_and_tokenizer(model_name)
    print(f"Model loaded: {type(model).__name__}", flush=True)

    print("Preparing inputs...", flush=True)
    input_args, formmatted_prompts = _prepare_inputs(
        [DEFAULT_PROMPT], tokenizer, model.config, BATCH_SIZE, MAX_LENGTH
    )
    max_tokens_to_generate = 20
    print(f"Input IDs shape: {input_args['input_ids'].shape}", flush=True)

    print("Transferring model and inputs to device...", flush=True)
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
        if isinstance(getattr(layer, "cumulative_length", None), torch.Tensor):
            layer.cumulative_length = layer.cumulative_length.to(device)
        if hasattr(layer, "device"):
            layer.device = device

    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)
    print("Transfer complete.", flush=True)

    print("Marking sharding on inputs and model...", flush=True)
    mark_sharding_on_inputs_and_model(model, input_args, mesh)
    print("Sharding marked.", flush=True)

    print("Compiling model...", flush=True)
    torch_xla.set_custom_compile_options({"experimental_weight_dtype": "bfp_bf8"})
    compiled_model = torch.compile(model, backend="tt")
    print("Model compiled.", flush=True)
    print(
        f"Starting generation loop (max {max_tokens_to_generate} tokens)...", flush=True
    )
    output_tokens: List[List[str]] = [[] for _ in range(1)]
    with torch.no_grad():
        for step in range(max_tokens_to_generate):
            output = compiled_model(**input_args)
            print(
                f"[Step {step}] {'Prefill' if step == 0 else 'Decode'} ...", flush=True
            )
            logits = output.logits.to("cpu")
            next_token_id = logits[:, -1].argmax(dim=-1)
            output_text = [tokenizer.decode(next_token_id[0])]
            for i, output_tokens_list in enumerate(output_tokens):
                output_tokens_list.append(output_text[i])

            if torch.all(next_token_id == tokenizer.eos_token_id):
                print()  # Add newline after generation completes
                break

            # Update inputs for next iteration
            input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)

    print()
    for i in range(1):
        print(f"=" * 80)
        print(f"Result for user {i}:")
        print(f"-" * 80)
        print("PROMPT:")
        print(formmatted_prompts[i])
        print(f"-" * 80)
        print("GENERATED:")
        print("".join(output_tokens[i]))
        print(f"=" * 80)
        print()


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
    elif num_devices == 4:  # 4-device host: batch axis 1, model-parallel axis 4
        mesh_shape = (1, 4)
    else:
        raise RuntimeError(
            f"Gemma4 31B expects 4, 8 (llmbox), or 32 (Galaxy) devices, "
            f"got {num_devices} devices"
        )

    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")
    return mesh


def mark_sharding_on_inputs_and_model(
    model: torch.nn.Module, input_args: dict, mesh: Mesh
):
    """
    Mark sharding on inputs and model weights for Gemma4 31B.

    Uses a 2D mesh ("batch", "model") matching the physical device topology.

    Gemma4 has two attention layer types with different KV head counts:
      - full_attention: 4 KV heads, global_head_dim=512, v_proj=None (k_eq_v)
      - sliding_attention: 16 KV heads, head_dim=256

    Args:
        model: Gemma4ForCausalLM instance (already on device)
        input_args: Input arguments dictionary containing past_key_values
        mesh: Device mesh with ("batch", "model") axes
    """
    # Gemma4ForCausalLM → .model (Gemma4TextModel)
    text_model = model.model

    # KV cache: shard on heads dim
    for layer in input_args["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    # Embedding and output head
    xs.mark_sharding(text_model.embed_tokens.weight, mesh, (None, "model"))
    xs.mark_sharding(text_model.norm.weight, mesh, (None,))
    xs.mark_sharding(model.lm_head.weight, mesh, ("model", None))

    for layer in text_model.layers:
        # Attention: column-parallel q/k/v, row-parallel o
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        if layer.self_attn.v_proj is not None:
            xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))

        # MLP: column-parallel gate/up, row-parallel down
        xs.mark_sharding(layer.mlp.gate_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.up_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.mlp.down_proj.weight, mesh, (None, "model"))

        # Layer norms (replicated across both mesh axes)
        xs.mark_sharding(layer.input_layernorm.weight, mesh, (None,))
        xs.mark_sharding(layer.post_attention_layernorm.weight, mesh, (None,))
        xs.mark_sharding(layer.pre_feedforward_layernorm.weight, mesh, (None,))
        xs.mark_sharding(layer.post_feedforward_layernorm.weight, mesh, (None,))


def _load_model_and_tokenizer(model_name: str):
    # Load as Gemma4ForConditionalGeneration (matches checkpoint key layout),
    # then extract text-only parts into a Gemma4ForCausalLM wrapper to avoid
    # the multimodal forward path that breaks under SPMD.
    full_model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    config = full_model.config
    model = Gemma4ForCausalLM(config.text_config)
    model.model = full_model.model.language_model
    model.lm_head = full_model.lm_head
    del full_model

    override_gemma4_sliding_window_causal_mask()
    model = model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _config_for_static_cache(model_config: PretrainedConfig) -> PretrainedConfig:
    """
    Shallow copy of decoder config for ``StaticCache`` only.

    When ``num_kv_shared_layers == 0``, transformers does ``layer_types[:-0]`` which is
    empty in Python. Setting ``num_kv_shared_layers = -num_hidden_layers`` makes that
    slice equivalent to ``layer_types[:num_hidden_layers]`` (full list). The model keeps
    the original config.
    """
    text = (
        model_config.get_text_config(decoder=True)
        if hasattr(model_config, "get_text_config")
        else model_config
    )
    cfg = copy.copy(text)
    if getattr(cfg, "num_kv_shared_layers", None) == 0:
        nh = getattr(cfg, "num_hidden_layers", None)
        if nh is not None:
            cfg.num_kv_shared_layers = -nh
    return cfg


def _prepare_inputs(
    input_prompt: List[str],
    tokenizer: PreTrainedTokenizer,
    model_config: PretrainedConfig,
    batch_size: int,
    max_cache_len: int,
) -> tuple[dict, List[str]]:

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
    seq_len = inputs.input_ids.shape[1]

    static_cache = StaticCache(
        config=_config_for_static_cache(model_config),
        max_cache_len=max_cache_len,
    )
    _init_static_cache(static_cache, model_config, BATCH_SIZE)

    sliding_window = getattr(
        model_config.get_text_config(decoder=True), "sliding_window", max_cache_len
    )
    # Override the cache sliding window layers to use the torch.compile/TT-friendly version
    override_cache_sliding_window_layers(static_cache, max_cache_len, sliding_window)

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
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }

    return input_args, formatted_prompts


def _init_static_cache(
    cache, config, batch_size, dtype=torch.bfloat16, device=torch.device("cpu")
):
    """Per-layer cache init for models with mixed head dimensions (e.g. Gemma4)."""
    text_config = (
        config.get_text_config(decoder=True)
        if hasattr(config, "get_text_config")
        else config
    )
    layer_types = getattr(
        text_config, "layer_types", ["full_attention"] * len(cache.layers)
    )

    nkv = getattr(text_config, "num_kv_shared_layers", 0)
    num_hidden = getattr(text_config, "num_hidden_layers", None)
    if num_hidden is not None and nkv > 0 and len(layer_types) == num_hidden:
        layer_types = layer_types[:-nkv]

    if len(layer_types) != len(cache.layers):
        raise ValueError(
            f"layer_types length ({len(layer_types)}) != len(cache.layers) "
            f"({len(cache.layers)}); check layer_types / num_kv_shared_layers."
        )

    for layer, layer_type in zip(cache.layers, layer_types):
        if layer_type == "full_attention" and getattr(
            text_config, "global_head_dim", None
        ):
            hd = text_config.global_head_dim
            nh = (
                getattr(text_config, "num_global_key_value_heads", None)
                or text_config.num_key_value_heads
            )
        else:
            hd = text_config.head_dim
            nh = text_config.num_key_value_heads

        fake_kv = torch.zeros((batch_size, nh, 0, hd), dtype=dtype, device=device)
        layer.lazy_initialization(fake_kv, fake_kv)


if __name__ == "__main__":
    xr.set_device_type("TT")
    gemma_4_31b()
