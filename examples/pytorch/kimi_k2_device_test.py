# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simplified Kimi K2 device-only test.
Runs prefill or decode on TT device without CPU comparison.

Usage:
    pytest kimi_k2_device_test.py -k prefill -s
    pytest kimi_k2_device_test.py -k decode -s
    NUM_LAYERS=2 BATCH_SIZE=64 pytest kimi_k2_device_test.py -k prefill -s
"""

import os

import numpy as np
import pytest
import setproctitle
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

setproctitle.setproctitle("kimi")


# ============== CONFIGURATION ==============
# Configure via environment variables:
#   NUM_LAYERS=2 BATCH_SIZE=64 pytest kimi_k2_device_test.py -k prefill -s

DEFAULT_NUM_LAYERS = 2
DEFAULT_BATCH_SIZE = 64
DEFAULT_INPUT_SEQ_LEN = 128
DEFAULT_MAX_OUTPUT_TOKENS = 1


# ============git== PYTEST FIXTURES ==============


@pytest.fixture
def num_layers():
    return int(os.environ.get("NUM_LAYERS", DEFAULT_NUM_LAYERS))


@pytest.fixture
def batch_size():
    return int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))


@pytest.fixture
def input_seq_len():
    return int(os.environ.get("INPUT_SEQ_LEN", DEFAULT_INPUT_SEQ_LEN))


@pytest.fixture
def max_output_tokens():
    return int(os.environ.get("MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))


# ============== SETUP FUNCTIONS ==============


def setup_spmd():
    """Initialize SPMD mode for multi-device."""
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()


def _kimi_k2_mesh_config(self, num_devices):
    """Mesh config supporting 64/32/8 devices."""
    if num_devices == 64:
        mesh_shape = (4, 16)
    elif num_devices == 32:  # Galaxy
        mesh_shape = (4, 8)
    elif num_devices == 8:  # llmbox
        mesh_shape = (2, 4)
    else:
        raise ValueError(f"Kimi K2: unsupported num_devices={num_devices}")
    return mesh_shape, ("batch", "model")


def load_model_and_tokenizer(num_layers: int):
    """Load Kimi K2 model with specified number of layers."""
    from third_party.tt_forge_models.kimi_k2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Monkey-patch mesh config
    ModelLoader.get_mesh_config = _kimi_k2_mesh_config

    model_loader = ModelLoader(
        variant=ModelVariant.KIMI_K2_INSTRUCT_MODIFIED, num_layers=num_layers
    )
    model = model_loader.load_model(dtype_override=torch.bfloat16)
    model = model.eval()
    tokenizer = model_loader.tokenizer

    return model, tokenizer, model_loader


def create_mesh(model_loader) -> Mesh:
    """Create device mesh from model loader config."""
    num_devices = xr.global_runtime_device_count()
    mesh_shape, mesh_names = model_loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, mesh_names)


def apply_sharding(model, model_loader, mesh, replicate_unsharded: bool = True):
    """Apply sharding specs to model weights.

    Args:
        model: The model to shard.
        model_loader: Model loader with load_shard_spec method.
        mesh: The device mesh.
        replicate_unsharded: If True, mark unsharded tensors with replicated
            sharding (all None) to ensure they are transferred to device.
    """
    import gc

    shard_specs = model_loader.load_shard_spec(model)

    # Track which tensors get sharded
    all_tensors = {}
    for name, p in model.named_parameters():
        all_tensors[id(p)] = ("param", name, p)
    for name, b in model.named_buffers():
        all_tensors[id(b)] = ("buffer", name, b)

    sharded_ids = set()
    for tensor, spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, spec)
        sharded_ids.add(id(tensor))

    # Find unsharded tensors
    unsharded = [
        (kind, name, tensor)
        for tid, (kind, name, tensor) in all_tensors.items()
        if tid not in sharded_ids
    ]

    # Apply replicated sharding to unsharded tensors
    if replicate_unsharded and unsharded:
        print(f"\nApplying replicated sharding to {len(unsharded)} unsharded tensors:")
        for kind, name, tensor in sorted(unsharded, key=lambda x: x[1]):
            # Replicated sharding: all None for each dimension
            replicated_spec = tuple(None for _ in tensor.shape)
            xs.mark_sharding(tensor, mesh, replicated_spec)
            print(f"  [{kind}] {name} -> {replicated_spec}")
        sharded_ids.update(id(t) for _, _, t in unsharded)
    elif unsharded:
        print(f"\nUnsharded tensors ({len(unsharded)}):")
        for kind, name, _ in sorted(unsharded, key=lambda x: x[1]):
            print(f"  [{kind}] {name}")

    print(
        f"\nSharding summary: {len(all_tensors)} total, " f"{len(sharded_ids)} sharded"
    )

    # Clear local references and run GC
    del all_tensors, unsharded, shard_specs
    gc.collect()
    print("CPU tensor storage cleared and GC completed.")


def init_mla_cache(config, batch_size: int, max_cache_len: int):
    """Initialize MLA cache for Kimi K2."""
    from infra import MLACache

    cache = MLACache(config=config, max_cache_len=max_cache_len)
    text_config = config.get_text_config(decoder=True)
    kv_lora_rank = text_config.kv_lora_rank
    qk_rope_head_dim = text_config.qk_rope_head_dim

    dummy_kv = torch.zeros((batch_size, 1, 1, kv_lora_rank), dtype=torch.bfloat16)
    dummy_pe = torch.zeros((batch_size, 1, 1, qk_rope_head_dim), dtype=torch.bfloat16)

    for layer in cache.layers:
        layer.lazy_initialization(dummy_kv, dummy_pe)

    return cache


def transfer_cache_to_device(cache, device):
    """Transfer MLA cache tensors to device."""
    for layer in cache.layers:
        layer.compressed_kv = layer.compressed_kv.to(device)
        layer.k_pe = layer.k_pe.to(device)
        layer.keys = layer.compressed_kv
        layer.values = layer.k_pe
        torch._dynamo.mark_static_address(layer.compressed_kv)
        torch._dynamo.mark_static_address(layer.k_pe)


def shard_cache(cache, mesh):
    """Apply sharding to MLA cache."""
    kv_spec = ("batch", None, None, None)
    for layer in cache.layers:
        xs.mark_sharding(layer.compressed_kv, mesh, kv_spec)
        xs.mark_sharding(layer.k_pe, mesh, kv_spec)


def construct_inputs(tokenizer, batch_size: int, seq_len: int, cache):
    """Construct input_ids and cache_position."""
    prompt = "Here is an exhaustive list of the best practices for writing clean code:"
    inputs = tokenizer(
        [prompt] * batch_size,
        return_tensors="pt",
        max_length=seq_len,
        truncation=True,
        padding="max_length",
    )
    cache_position = torch.arange(0, inputs.input_ids.shape[1])

    return {
        "input_ids": inputs.input_ids,
        "past_key_values": cache,
        "cache_position": cache_position,
        "use_cache": True,
    }


def extract_and_print_tokens(logits, tokenizer, prefix=""):
    """Argmax logits and print decoded tokens."""
    predicted_ids = logits[:, -1].argmax(dim=-1)
    decoded = tokenizer.batch_decode(predicted_ids.cpu())
    print(f"\n{prefix}Predicted tokens:")
    for i, text in enumerate(decoded):
        print(f"  User {i}: {repr(text)}")


# ============== TESTS ==============


def test_kimi_k2_prefill(num_layers, batch_size, input_seq_len):
    """Run prefill on device, print argmaxed tokens."""
    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()

    # Load model
    print(f"\nLoading Kimi K2 with {num_layers} layers...")
    model, tokenizer, model_loader = load_model_and_tokenizer(num_layers)

    # Create mesh and apply sharding
    mesh = create_mesh(model_loader)
    print(f"Mesh: {mesh.shape()}")

    model = model.to(device)
    apply_sharding(model, model_loader, mesh)

    # Create inputs
    cache = init_mla_cache(model.config, batch_size, input_seq_len)
    input_args = construct_inputs(tokenizer, batch_size, input_seq_len, cache)

    # Transfer to device
    transfer_cache_to_device(input_args["past_key_values"], device)
    shard_cache(input_args["past_key_values"], mesh)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))

    # Set compile options (including experimental_weight_dtype)
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": 0,  # Minimal optimization for stability
            "enable_trace": False,  # Disabled due to topk indices issue
            "experimental_weight_dtype": "bfp_bf8",
            # Migrate const-eval inputs (the weights) to device DRAM on first
            # use instead of pinning them in host system memory for the buffer's
            # lifetime. Releases the ~38 GB host staging at the cost of more
            # device DRAM pressure.
            "enable_const_eval_inputs_to_system_memory": False,
        }
    )

    # Compile and run
    print("Compiling model...")
    compiled_model = torch.compile(model, backend="tt")

    print("Running prefill...")
    with torch.no_grad():
        output = compiled_model(**input_args)

    logits = output.logits.cpu()
    extract_and_print_tokens(logits, tokenizer, prefix="[PREFILL] ")


def test_kimi_k2_decode(num_layers, batch_size, input_seq_len, max_output_tokens):
    """Run decode step(s) on device with empty cache, print argmaxed tokens."""
    xr.set_device_type("TT")
    setup_spmd()
    device = torch_xla.device()

    # Load model
    print(f"\nLoading Kimi K2 with {num_layers} layers...")
    model, tokenizer, model_loader = load_model_and_tokenizer(num_layers)

    # Create mesh and apply sharding
    mesh = create_mesh(model_loader)
    print(f"Mesh: {mesh.shape()}")

    model = model.to(device)
    apply_sharding(model, model_loader, mesh)

    # Create inputs - single token input for decode
    cache = init_mla_cache(model.config, batch_size, input_seq_len)

    # Use a single token as input (decode step)
    single_token = tokenizer.encode("Hello", return_tensors="pt")[:, :1]
    input_ids = single_token.expand(batch_size, -1).contiguous()
    cache_position = torch.tensor([0])  # Start of sequence

    input_args = {
        "input_ids": input_ids,
        "past_key_values": cache,
        "cache_position": cache_position,
        "use_cache": True,
    }

    # Transfer to device
    transfer_cache_to_device(input_args["past_key_values"], device)
    shard_cache(input_args["past_key_values"], mesh)
    input_args["input_ids"] = input_args["input_ids"].to(device)
    input_args["cache_position"] = input_args["cache_position"].to(device)
    xs.mark_sharding(input_args["input_ids"], mesh, ("batch", None))

    # Set compile options (including experimental_weight_dtype)
    torch_xla.set_custom_compile_options(
        {
            "optimization_level": 0,  # Minimal optimization for stability
            "enable_trace": False,  # Disabled due to topk indices issue
            "experimental_weight_dtype": "bfp_bf8",
            # Migrate const-eval inputs (the weights) to device DRAM on first
            # use instead of pinning them in host system memory for the buffer's
            # lifetime. Releases the ~38 GB host staging at the cost of more
            # device DRAM pressure.
            "enable_const_eval_inputs_to_system_memory": False,
        }
    )

    # Compile and run
    print("Compiling model...")
    compiled_model = torch.compile(model, backend="tt")

    print(f"Running {max_output_tokens} decode step(s)...")
    with torch.no_grad():
        for step in range(max_output_tokens):
            output = compiled_model(**input_args)
            logits = output.logits.cpu()

            extract_and_print_tokens(logits, tokenizer, prefix=f"[DECODE step {step}] ")

            # Update for next step
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_args["input_ids"] = next_token.to(device)
            input_args["cache_position"] = input_args["cache_position"][-1:] + 1
