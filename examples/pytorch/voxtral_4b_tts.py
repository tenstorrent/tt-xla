# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Tensor-parallel forward pass of the Voxtral-4B-TTS LM backbone.

Voxtral-4B-TTS is a Mistral-native text-to-speech model. Its compute-dominant
component is a 26-layer Mistral decoder LM backbone that autoregressively
predicts audio-codebook tokens; the acoustic head and neural codec that turn
those tokens into a 24 kHz waveform are TTS-specific and are not part of this
example. This script runs the backbone as a single Megatron tensor-parallel
forward pass over a text prompt, sharded across the available chips.

Everything — the model, its inputs, the TP mesh, and the shard plan — is driven
through the tt-forge-models ``ModelLoader`` public API, so this doubles as
reference code for the loader. Modelled after ``qwen3_tp.py``.
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
    # Enable SPMD / Shardy and discover the device count.
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    num_devices = xr.global_runtime_device_count()

    # Build the backbone and its inputs entirely through the loader.
    loader = ModelLoader(ModelVariant.VOXTRAL_4B_TTS)
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    # Megatron tensor-parallel mesh (1 x N model axis) from the loader.
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    # Move inputs and model to the TT device.
    device = torch_xla.device()
    inputs = {k: v.to(device) for k, v in inputs.items()}
    model = model.to(device)

    # Apply the loader's Megatron column->row shard plan.
    for tensor, shard_spec in loader.load_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    # Compile and run.
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    logits = output.logits.cpu()
    return logits, inputs["input_ids"].cpu(), loader, model.config.vocab_size


def post_process_output(logits, input_ids, loader):
    """Print the prompt the backbone saw and its next-token prediction.

    The backbone predicts audio-codebook positions rather than clean text, so
    the decoded argmax token is typically empty/non-textual — the meaningful,
    human-readable result here is the reconstructed prompt and the shape/health
    of the produced logits. The loader's public ``tokenizer`` (populated by
    ``load_inputs``) reconstructs the prompt.
    """
    # Strip right padding (token id 0) to recover the real prompt tokens.
    ids = input_ids[0].tolist()
    prompt_ids = [i for i in ids if i != 0]
    prompt = loader.tokenizer.decode(prompt_ids)

    next_id = int(logits[0, -1].argmax())
    next_tok = loader.tokenizer.decode([next_id])

    print(f"Prompt: {prompt!r}")
    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Next-token argmax id: {next_id}  decoded: {next_tok!r}")
    return prompt


def test_voxtral_4b_tts():
    """Guard the example: the TP backbone forward must produce finite logits of
    the expected ``(1, seq_len, vocab_size)`` shape."""
    xr.set_device_type("TT")

    logits, input_ids, loader, vocab_size = voxtral_4b_tts()

    seq_len = input_ids.shape[1]
    assert logits.shape == (1, seq_len, vocab_size), (
        f"Unexpected logits shape {tuple(logits.shape)}; "
        f"expected {(1, seq_len, vocab_size)}"
    )
    assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"

    post_process_output(logits, input_ids, loader)
    print("Voxtral-4B-TTS TP backbone forward produced finite logits.")


if __name__ == "__main__":
    # By default torch_xla uses the CPU device so we have to set it to TT device.
    xr.set_device_type("TT")

    logits, input_ids, loader, _ = voxtral_4b_tts()
    post_process_output(logits, input_ids, loader)
