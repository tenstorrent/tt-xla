# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
GPT-OSS 120B test on QB2 (4 chips, 1x4 mesh).

Runs the model the same way as examples/pytorch/gpt_oss_120b.py:
- MXFP4 quantized weights (dequantized to bf16)
- Mixed precision: bfp_bf8 default + bfp_bf4 for expert weights
- Sparse MoE with cluster_axis=1
- StaticCache with PREFILL_PAD_LEN=100
- Full sharding matching the examples path
"""

import os

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import StaticCache

from tt_torch.sparse_mlp import A2aSparseMLP, enable_sparse_mlp
from tt_torch.weight_dtype import apply_weight_dtype_overrides

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


BATCH_SIZE = 8
MAX_CACHE_LEN = 256
PREFILL_PAD_LEN = 100  # Must NOT be a multiple of 32 (avoids split_seq dispatch path)


def setup_spmd():
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def create_device_mesh():
    num_devices = xr.global_runtime_device_count()
    if num_devices == 32:
        mesh_shape = (4, 8)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    elif num_devices == 4:
        mesh_shape = (1, 4)
    else:
        pytest.skip(f"GPT-OSS 120B requires 4, 8, or 32 devices (got {num_devices})")
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    return mesh, mesh_shape


def construct_inputs(tokenizer, model_config, batch_size, max_cache_len):
    messages = [{"role": "user", "content": "Who are you?"}]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompts = [formatted] * batch_size

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=PREFILL_PAD_LEN,
        padding="max_length",
        padding_side="left",
        return_attention_mask=True,
    )

    # Disable sliding window to avoid recompilation
    model_config.layer_types = ["full_attention"] * model_config.num_hidden_layers

    static_cache = StaticCache(
        config=model_config,
        max_batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )
    num_kv_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim
    static_cache.early_initialization(
        batch_size=batch_size,
        num_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    prompt_len = inputs.input_ids.shape[1]
    full_attention_mask = torch.ones(
        (batch_size, max_cache_len), dtype=inputs.attention_mask.dtype
    )
    full_attention_mask[:, :prompt_len] = inputs.attention_mask

    return {
        "input_ids": inputs.input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": full_attention_mask,
    }


def transfer_to_device(model, input_args, device):
    for layer in input_args["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    input_args["attention_mask"] = input_args["attention_mask"].to(device)
    model = model.to(device)
    return model, input_args


def mark_sharding(model, input_args, mesh, sparse_moe=True):
    for layer in input_args["past_key_values"].layers:
        xs.mark_sharding(layer.keys, mesh, (None, "model", None, None))
        xs.mark_sharding(layer.values, mesh, (None, "model", None, None))

    xs.mark_sharding(model.model.embed_tokens.weight, mesh, (None, None))
    xs.mark_sharding(model.model.norm.weight, mesh, (None,))

    for layer in model.model.layers:
        xs.mark_sharding(layer.self_attn.q_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.k_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.v_proj.weight, mesh, ("model", None))
        xs.mark_sharding(layer.self_attn.o_proj.weight, mesh, (None, "model"))
        xs.mark_sharding(layer.self_attn.sinks, mesh, (None,))

        if sparse_moe and isinstance(layer.mlp, A2aSparseMLP):
            expert_e_shard = (("batch", "model"), None, None)
            expert_e_bias_shard = (("batch", "model"), None)
        else:
            expert_e_shard = ("model", None, None)
            expert_e_bias_shard = ("model", None)
        xs.mark_sharding(layer.mlp.experts.gate_up_proj, mesh, expert_e_shard)
        xs.mark_sharding(layer.mlp.experts.gate_up_proj_bias, mesh, expert_e_bias_shard)
        xs.mark_sharding(layer.mlp.experts.down_proj, mesh, expert_e_shard)
        xs.mark_sharding(layer.mlp.experts.down_proj_bias, mesh, expert_e_bias_shard)

@pytest.mark.nightly
def test_gpt_oss_120b_prefill():
    """Test prefill step of GPT-OSS 120B with mixed precision (bf8+bf4) and sparse MoE."""
    xr.set_device_type("TT")
    setup_spmd()

    device = torch_xla.device()
    mesh, mesh_shape = create_device_mesh()

    # Load model via loader (uses MXFP4 quantization, bf16, eager attention)
    loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B)
    model = loader.load_model()
    tokenizer = loader.tokenizer

    # Apply mixed precision: bf8 default + bf4 for expert weights
    applied = apply_weight_dtype_overrides(
        model,
        {
            "default": "bfp_bf8",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
    )
    print(f"Applied {len(applied)} weight dtype overrides")

    # Enable sparse MoE
    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=1)

    # Construct inputs with static cache
    input_args = construct_inputs(tokenizer, model.config, BATCH_SIZE, MAX_CACHE_LEN)

    # Transfer to device
    model, input_args = transfer_to_device(model, input_args, device)

    # Apply sharding
    mark_sharding(model, input_args, mesh, sparse_moe=True)

    # Compile and run prefill
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**input_args)
        output_logits = output.logits.to("cpu")

    print(f"Output logits shape: {output_logits.shape}")
    next_token_id = output_logits[:, -1].argmax(dim=-1)
    decoded = tokenizer.decode(next_token_id[0].item(), skip_special_tokens=True)
    print(f"Next token: {decoded}")

    assert output_logits.shape[0] == BATCH_SIZE
    assert output_logits.shape[1] == PREFILL_PAD_LEN
    print("Prefill test passed.")

@pytest.mark.nightly
def test_gpt_oss_120b_decode():
    """Test prefill + one decode step with mixed precision (bf8+bf4) and sparse MoE."""
    xr.set_device_type("TT")
    setup_spmd()
    
    device = torch_xla.device()
    mesh, mesh_shape = create_device_mesh()

    loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B)
    model = loader.load_model()
    tokenizer = loader.tokenizer

    applied = apply_weight_dtype_overrides(
        model,
        {
            "default": "bfp_bf8",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
    )
    print(f"Applied {len(applied)} weight dtype overrides")

    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=1)

    input_args = construct_inputs(tokenizer, model.config, BATCH_SIZE, MAX_CACHE_LEN)
    model, input_args = transfer_to_device(model, input_args, device)
    mark_sharding(model, input_args, mesh, sparse_moe=True)

    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        # Prefill
        output = compiled_model(**input_args)
        output_logits = output.logits.to("cpu")
        next_token_id = output_logits[:, -1].argmax(dim=-1)
        print(f"Prefill done. Next token: {tokenizer.decode(next_token_id[0].item())}")

        # Decode step
        input_args["input_ids"] = next_token_id.unsqueeze(-1).to(device)
        host_cache_pos = input_args["cache_position"].to("cpu")
        host_cache_pos = torch.tensor([host_cache_pos[-1:] + 1])
        input_args["cache_position"] = host_cache_pos.to(device)

        decode_output = compiled_model(**input_args)
        decode_logits = decode_output.logits.to("cpu")

    decode_token_id = decode_logits[:, -1].argmax(dim=-1)
    decoded = tokenizer.decode(decode_token_id[0].item(), skip_special_tokens=True)
    print(f"Decode token: {decoded}")

    assert decode_logits.shape[0] == BATCH_SIZE
    assert decode_logits.shape[1] == 1
    print("Prefill + decode test passed.")


@pytest.mark.nightly
def test_gpt_oss_120b_decode_only():
    """Test decode-only step (single token, no prefill) with mixed precision (bf8+bf4) and sparse MoE."""
    xr.set_device_type("TT")
    setup_spmd()

    device = torch_xla.device()
    mesh, mesh_shape = create_device_mesh()

    loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B)
    model = loader.load_model()
    tokenizer = loader.tokenizer

    applied = apply_weight_dtype_overrides(
        model,
        {
            "default": "bfp_bf8",
            "model.layers.*.mlp.experts.gate_up_proj": "bfp_bf4",
            "model.layers.*.mlp.experts.down_proj": "bfp_bf4",
        },
    )
    print(f"Applied {len(applied)} weight dtype overrides")

    enable_sparse_mlp(model, mesh=mesh_shape, cluster_axis=1)

    # Disable sliding window to avoid recompilation
    model.config.layer_types = ["full_attention"] * model.config.num_hidden_layers

    # Build decode-only inputs: single token, cache_position=[0], empty static cache
    static_cache = StaticCache(
        config=model.config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=MAX_CACHE_LEN,
        device="cpu",
        dtype=torch.bfloat16,
    )
    static_cache.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        dtype=torch.bfloat16,
        device="cpu",
    )

    # Single token input (decode shape: batch x 1)
    input_ids = torch.randint(0, model.config.vocab_size, (BATCH_SIZE, 1))
    attention_mask = torch.ones((BATCH_SIZE, MAX_CACHE_LEN), dtype=torch.long)
    cache_position = torch.tensor([0])

    input_args = {
        "input_ids": input_ids,
        "past_key_values": static_cache,
        "cache_position": cache_position,
        "use_cache": True,
        "attention_mask": attention_mask,
    }

    # Transfer to device
    model, input_args = transfer_to_device(model, input_args, device)
    mark_sharding(model, input_args, mesh, sparse_moe=True)

    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**input_args)
        output_logits = output.logits.to("cpu")

    print(f"Decode output shape: {output_logits.shape}")
    next_token_id = output_logits[:, -1].argmax(dim=-1)
    decoded = tokenizer.decode(next_token_id[0].item(), skip_special_tokens=True)
    print(f"Decode token: {decoded}")

    assert output_logits.shape[0] == BATCH_SIZE
    assert output_logits.shape[1] == 1
    print("Decode-only test passed.")
