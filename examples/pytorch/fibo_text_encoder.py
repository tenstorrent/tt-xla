# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
FIBO (briaai/FIBO) text-encoder — tensor-parallel (TP-4) inference example.

FIBO is BRIA AI's 8B-parameter DiT text-to-image model. It conditions its DiT on
the transformer hidden states of its text encoder — ``SmolLM3-3B`` — discarding
the vocabulary head. This example runs *only* that text encoder end-to-end on a
4-chip Tenstorrent device using Megatron-1D tensor parallelism (TP-4), producing
the ``last_hidden_state`` conditioning embedding the DiT would consume.

Everything comes from the tt-forge-models loader's public API:
  * ``load_model`` / ``load_inputs`` — weights + a tokenized structured caption.
  * ``get_mesh_config`` — the ``(1, num_devices)`` ``("batch", "model")`` mesh
    (TP-4 on a 4-chip part; no data parallelism).
  * ``load_shard_spec`` — the Megatron-1D (column→row) parameter partitioning:
    attention Q/K/V and MLP gate/up are column-parallel, attention-out and MLP
    down are row-parallel, everything else replicated. GQA is preserved (TP-4:
    4 query heads + 1 KV head per chip).

Context length: SmolLM3-3B supports up to ``MAX_CONTEXT_LENGTH`` (65536)
positions architecturally, but on a 4-chip TP-4 device the largest context that
compiles, runs, and stays within per-chip DRAM is ``DEFAULT_CONTEXT_LENGTH``
(24576) — larger values OOM at runtime (see the loader docstring). This example
therefore runs at that TP-4 ceiling, batch size 1. Override via
``FIBO_TE_CONTEXT_LENGTH`` (more chips are needed for longer contexts).

Modelled after ``qwen3_tp.py`` (the SPMD tensor-parallel example), but driven
through the loader's mesh/shard-spec hooks instead of a hand-written spec.

Note: ``briaai/FIBO`` is gated — accept its license and set ``HF_TOKEN``.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.fibo.text_encoder.pytorch.loader import (
    DEFAULT_CONTEXT_LENGTH,
    MAX_CONTEXT_LENGTH,
    ModelLoader,
    ModelVariant,
)

# Run at the largest context the 4-chip TP-4 topology supports. The loader
# clamps to the architectural max (MAX_CONTEXT_LENGTH); DEFAULT_CONTEXT_LENGTH is
# the validated TP-4 runtime ceiling. Respect a caller-provided override.
os.environ.setdefault("FIBO_TE_CONTEXT_LENGTH", str(DEFAULT_CONTEXT_LENGTH))


def fibo_text_encoder():
    """Compile and run the FIBO text encoder with TP-4 on device.

    Returns:
        tuple: ``(hidden_states, prompt_preview)`` — the ``last_hidden_state``
        conditioning embedding of shape ``[1, context_length, hidden_size]``
        (moved back to CPU as float32) and a decoded preview of the caption the
        loader actually fed the encoder.
    """
    # Enable SPMD + Shardy: the tt-mlir stablehlo pipeline expects shardy
    # annotations on the incoming graph (same setup the runner uses for TP).
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the encoder + inputs via the loader (bf16 weights, as the runner uses).
    loader = ModelLoader(ModelVariant.BASE)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16, batch_size=1)

    # Decode the first tokens of the real input for a human-readable preview
    # (loader.tokenizer is populated by load_inputs; a structured JSON caption).
    prompt_preview = loader.tokenizer.decode(inputs["input_ids"][0, :48])

    # Mesh from the loader: (1, num_devices) over ("batch", "model") = TP-4.
    mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
    mesh = Mesh(np.array(range(num_devices)), mesh_shape, mesh_names)
    print(
        f"Mesh {mesh_shape} {mesh_names} over {num_devices} devices (TP-{mesh_shape[1]})."
    )

    # Move model + inputs to the TT device.
    device = torch_xla.device()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Apply the loader's Megatron-1D shard spec (one all-reduce per attn / MLP).
    for param, partition_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(param, mesh, partition_spec)

    # Compile with the "tt" backend. No custom compile options: the FIBO TP config
    # is optimization_level 0 (the default), matching the bringup gate.
    compiled_model = torch.compile(model, backend="tt")

    with torch.no_grad():
        output = compiled_model(**inputs)

    return output.last_hidden_state.to(torch.float32).cpu(), prompt_preview


def post_process_output(hidden_states, prompt_preview):
    """Print a human-readable summary of the conditioning embedding.

    The text encoder emits hidden states (no vocabulary head / text to decode),
    so the real-world result is the embedding tensor the DiT conditions on: its
    shape, finiteness, and magnitude statistics, plus per-token norms for the
    first tokens of the caption.
    """
    _, seq, hidden = hidden_states.shape
    norms = hidden_states.norm(dim=-1)  # [batch, seq] per-token L2 norm

    print("\n=== FIBO text-encoder conditioning embedding ===")
    print(
        f"Caption fed to encoder (first tokens, tiled to fill context):\n  {prompt_preview}"
    )
    print(
        f"last_hidden_state shape : {tuple(hidden_states.shape)}  (batch, seq, hidden)"
    )
    print(f"context length          : {seq}")
    print(f"hidden size             : {hidden}")
    print(f"all finite              : {bool(torch.isfinite(hidden_states).all())}")
    print(
        f"mean / std              : {hidden_states.mean():.4f} / {hidden_states.std():.4f}"
    )
    print(f"per-token L2 norm (first 5 tokens): {norms[0, :5].tolist()}")
    print(f"embedding[0, 0, :8]     : {hidden_states[0, 0, :8].tolist()}")
    return hidden_states


def test_fibo_text_encoder():
    """Guard the example: the encoder returns a finite conditioning embedding of
    the expected ``[1, context_length, hidden_size]`` shape."""
    xr.set_device_type("TT")

    expected_ctx = min(
        int(os.environ.get("FIBO_TE_CONTEXT_LENGTH", DEFAULT_CONTEXT_LENGTH)),
        MAX_CONTEXT_LENGTH,
    )

    hidden_states, _ = fibo_text_encoder()

    assert hidden_states.ndim == 3, f"expected rank-3 output, got {hidden_states.shape}"
    assert hidden_states.shape[0] == 1, "batch size should be 1"
    assert (
        hidden_states.shape[1] == expected_ctx
    ), f"context length {hidden_states.shape[1]} != expected {expected_ctx}"
    assert torch.isfinite(hidden_states).all(), "conditioning embedding is non-finite"

    print(
        "\nFIBO text-encoder produced a finite conditioning embedding of the "
        f"expected shape {tuple(hidden_states.shape)}."
    )


if __name__ == "__main__":
    # torch_xla defaults to CPU, so select the TT device.
    xr.set_device_type("TT")

    hidden_states, prompt_preview = fibo_text_encoder()
    post_process_output(hidden_states, prompt_preview)
