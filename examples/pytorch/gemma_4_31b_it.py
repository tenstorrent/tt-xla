# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Tensor-parallel (TP-4) single-prefill next-token example for google/gemma-4-31B-it.

google/gemma-4-31B-it is a 31B image-text-to-text VLM whose 30.7B text decoder
weighs ~62 GB in bfloat16 — too large for a single 32 GB Blackhole chip — so the
realistic scenario on a 4-chip part (qb2-blackhole) is Megatron-style 1D tensor
parallelism across the mesh. This example drives a single prefill forward over a
chat prompt and predicts the next token (greedy + top-5), which is the path the
model-bringup validated (logits PCC 0.9989 vs CPU). It deliberately does *not*
run a multi-token decode loop: the decode graph (use_cache=True) is a known
compiler gap for this model, so only the prefill / single-forward path is faithful.

Everything — model, tokenizer, inputs, mesh shape and the tensor-parallel shard
map — is driven through the tt-forge-models ``ModelLoader`` public API, mirroring
``examples/pytorch/qwen3_tp.py``'s SPMD ``mark_sharding`` structure.
"""

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.gemma4.pytorch import ModelLoader, ModelVariant


def gemma_4_31b_it():
    # Enable SPMD so the Shardy-based tensor-parallel path is used (same as qwen3_tp.py).
    import os

    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build model + tokenizer + inputs through the loader's public API. load_model
    # loads the text-only causal-LM path in eval mode with use_cache=False (prefill).
    loader = ModelLoader(ModelVariant.GEMMA_4_31B_IT)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs()  # {"input_ids", "attention_mask"} for the text path

    # Mesh shape + tensor-parallel shard map come from the loader (Megatron col->row,
    # KV replicated). get_mesh_config asserts num_attention_heads is divisible by the
    # model-axis size.
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)
    print(f"Created device mesh: {mesh_shape} with {num_devices} devices")

    # Move model + inputs to the TT device, then apply the loader's shard spec.
    device = torch_xla.device()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile + run a single prefill forward.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    logits = loader.unpack_forward_output(output).cpu().float()
    return logits, loader


def post_process_output(logits, loader):
    """Print the human-readable next-token prediction (greedy + top-5)."""
    tokenizer = loader.tokenizer
    next_token_logits = logits[0, -1]
    next_token_id = int(next_token_logits.argmax(dim=-1))
    next_token = tokenizer.decode([next_token_id])

    top5 = torch.topk(next_token_logits, k=5)
    print("=" * 80)
    print(f"Prompt: {loader.sample_text!r}")
    print("-" * 80)
    print(f"Predicted next token (greedy): id={next_token_id} -> {next_token!r}")
    print("Top-5 next-token candidates:")
    for rank, (tok_id, score) in enumerate(zip(top5.indices.tolist(), top5.values.tolist()), 1):
        print(f"  {rank}. id={tok_id:<7} {tokenizer.decode([tok_id])!r:<20} logit={score:.3f}")
    print("=" * 80)
    return next_token_id, next_token


def test_gemma_4_31b_it():
    """Guard: TP-4 prefill produces a finite, correctly-shaped logits tensor and a
    deterministic, decodable next token."""
    xr.set_device_type("TT")

    logits, loader = gemma_4_31b_it()

    # Vocab dimension must match the model config; logits must be finite.
    vocab_size = loader.config.get_text_config(decoder=True).vocab_size
    assert logits.shape[-1] == vocab_size, (
        f"Expected last dim {vocab_size}, got {logits.shape[-1]}"
    )
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"

    next_token_id, next_token = post_process_output(logits, loader)
    assert 0 <= next_token_id < vocab_size, "Predicted token id out of vocab range"
    assert next_token != "", "Predicted token decoded to an empty string"
    print(f"PASS: greedy next token id={next_token_id} -> {next_token!r}")


if __name__ == "__main__":
    # torch_xla defaults to CPU; select the TT device.
    xr.set_device_type("TT")

    logits, loader = gemma_4_31b_it()
    post_process_output(logits, loader)
