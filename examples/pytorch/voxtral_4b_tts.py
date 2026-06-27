# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Voxtral-4B-TTS LM-backbone tensor-parallel forward.

Voxtral-4B-TTS is a Mistral-native text-to-speech checkpoint. Its compute-dominant
component is a 26-layer Mistral decoder LM backbone that autoregressively predicts
audio-codebook tokens; the acoustic head and neural codec that turn those tokens
into a 24 kHz waveform are TTS-specific and are not part of this example.

This example drives that backbone through the tt-forge-models ``ModelLoader`` and
runs a single tensor-parallel (Megatron column->row) forward pass across the
device mesh, exactly the topology adopted as the model's bringup baseline. The
loader supplies the model, the tokenized inputs, the mesh shape, and the shard
plan, so the example contains no inline sharding logic -- it is reference code for
how to fan a loader-provided model out across an n300 with the loader's own hooks.
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


def voxtral_4b_tts():
    # Enable SPMD / Shardy and read the device count the mesh is built from.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the LM backbone and a tokenized prompt entirely through the loader.
    loader = ModelLoader(ModelVariant.VOXTRAL_4B_TTS)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    # Mesh + shard plan come from the loader so the example stays in lock-step
    # with the bringup baseline (a 1xN "model" axis, Megatron column->row).
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    # Move inputs and weights to device, then apply the loader's tensor-parallel
    # sharding to the on-device weights.
    device = torch_xla.device()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    vocab_size = model.config.vocab_size

    # Compile for the TT backend and run the forward pass.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    return loader, inputs, output.logits.cpu(), vocab_size


def post_process_output(loader, inputs, logits):
    """Print a human-readable view of the backbone forward pass."""
    # Reconstruct the prompt from the public tokenizer (populated by load_inputs).
    ids = inputs["input_ids"][0].cpu().tolist()
    # The loader right-pads with token id 0 to a static length; drop the padding
    # for display only.
    while len(ids) > 1 and ids[-1] == 0:
        ids.pop()
    prompt = loader.tokenizer.decode(ids)

    # Greedy next-token prediction at the last real position.
    next_token_id = int(logits[0, len(ids) - 1].argmax().item())

    print(f"Prompt: {prompt!r}")
    print(f"Logits shape: {tuple(logits.shape)} (batch, seq_len, vocab_size)")
    print(f"Vocab size: {logits.shape[-1]}")
    print(f"Argmax next-token id at last prompt position: {next_token_id}")
    # The backbone predicts audio-codebook positions rather than clean text, so a
    # decoded continuation is typically empty/non-textual -- shown for reference.
    print(f"Decoded next token: {loader.tokenizer.decode([next_token_id])!r}")


def test_voxtral_4b_tts():
    """Backbone TP forward produces a finite logits tensor of the right shape.

    Correctness here is shape + finiteness: this is the LM backbone of a TTS
    model, so its predictions land on audio-codebook token positions rather than
    readable text, and a decoded continuation carries no stable textual meaning.
    """
    xr.set_device_type("TT")

    loader, inputs, logits, vocab_size = voxtral_4b_tts()

    seq_len = inputs["input_ids"].shape[1]
    assert logits.shape == (1, seq_len, vocab_size), (
        f"Unexpected logits shape {tuple(logits.shape)}; "
        f"expected {(1, seq_len, vocab_size)}"
    )
    assert torch.isfinite(logits).all(), "Logits contain non-finite values"

    post_process_output(loader, inputs, logits)
    print("Voxtral-4B-TTS backbone TP forward produced finite logits.")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    loader, inputs, logits, _ = voxtral_4b_tts()
    post_process_output(loader, inputs, logits)
