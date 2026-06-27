# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Voxtral-4B-TTS LM backbone — tensor-parallel forward on n300.

Voxtral-4B-TTS is a Mistral-native text-to-speech checkpoint. Its compute-dominant
component is a 26-layer Mistral decoder LM backbone (Ministral-3-3B based) that
autoregressively predicts audio-codebook tokens; the acoustic-transformer head and
convolutional neural codec that turn those tokens into a 24 kHz waveform are
TTS-specific (conv weight-norm, data-dependent audio length) and are not brought up
on device.

This example demonstrates the realistic multi-chip scenario adopted at bringup: the
backbone is sharded **tensor-parallel** (Megatron column→row) across the two chips of
an n300 and run as a single forward pass over a tokenized prompt, producing the
next-token distribution over the model's 131072-entry vocabulary. Weights, inputs,
the TP mesh, and the shard plan all come from the tt-forge-models ``ModelLoader`` —
this script does not re-derive the architecture.

Modelled after ``examples/pytorch/qwen3_tp.py`` (single tensor-parallel forward via
``Mesh`` + ``xs.mark_sharding``), using the loader's ``get_mesh_config`` /
``load_shard_spec`` hooks instead of an inline shard plan.
"""

import os

import numpy as np
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from torch_xla.distributed.spmd import Mesh

from third_party.tt_forge_models.voxtral_tts.text_to_speech.pytorch import (
    ModelLoader,
    ModelVariant,
)


# --------------------------------
# Voxtral-4B-TTS tensor-parallel backbone example
# --------------------------------
def voxtral_4b_tts():
    # Enable SPMD + Shardy lowering (the path the TP bringup baseline was validated on).
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the LM backbone, inputs, and TP plan from the loader.
    loader = ModelLoader(ModelVariant.VOXTRAL_4B_TTS)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    # Tensor-parallel mesh (1 x num_devices "model" axis on n300).
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    # Move model + inputs to device.
    device = torch_xla.device()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Apply the loader's Megatron column->row shard plan.
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile and run a single forward pass.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    logits = output.logits.to("cpu")
    return logits, loader


def post_process_output(logits, loader):
    """Print the backbone's next-token prediction in a human-readable form."""
    # Next-token distribution at the final prompt position.
    next_token_logits = logits[0, -1]
    topk = torch.topk(next_token_logits.float(), k=5)

    print("=" * 80)
    print("Voxtral-4B-TTS LM backbone (tensor-parallel forward)")
    print("-" * 80)
    print(f"Prompt: {loader._SAMPLE_TEXT!r}")
    print(f"Logits shape: {tuple(logits.shape)}  (batch, seq_len, vocab)")
    print("-" * 80)
    print("Top-5 predicted next tokens:")
    for rank, (score, token_id) in enumerate(zip(topk.values.tolist(), topk.indices.tolist())):
        decoded = ""
        try:
            if loader.tokenizer is None:
                loader._load_tokenizer()
            decoded = repr(loader.tokenizer.decode([token_id]))
        except Exception:
            decoded = "<undecodable>"
        print(f"  {rank + 1}. id={token_id:<7} logit={score:8.3f}  {decoded}")
    print("=" * 80)


def test_voxtral_4b_tts():
    """The tensor-parallel backbone forward produces a well-formed, finite
    next-token distribution over the full vocabulary."""
    xr.set_device_type("TT")

    logits, loader = voxtral_4b_tts()

    # Expected shape: (batch=1, seq_len=32, vocab=131072).
    assert logits.shape[0] == 1, f"Unexpected batch dim: {logits.shape}"
    assert logits.shape[1] == loader._variant_config.max_length, (
        f"Unexpected seq_len: {logits.shape}"
    )
    assert logits.shape[2] == loader._VOCAB_SIZE, (
        f"Unexpected vocab dim: {logits.shape}"
    )
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    post_process_output(logits, loader)


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    logits, loader = voxtral_4b_tts()
    post_process_output(logits, loader)
