# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
XTTS-v2 text-to-speech GPT transformer-core inference example.

coqui/XTTS-v2 synthesizes speech by running an autoregressive GPT-2 transformer
that, conditioned on a speaker latent + text tokens, predicts discrete mel-code
logits; a separate HiFi-GAN vocoder then turns those codes into a waveform.

This example brings up the compute-dominant piece -- the 379M-parameter GPT-2
transformer core -- as a single static forward pass on a Tenstorrent device. The
speaker conditioning latent (a built-in speaker), text embeddings and mel start
tokens are precomputed on host by the tt-forge-models loader and fed in as
``inputs_embeds``; the device graph runs the transformer stack and emits the
mel-token logits that would feed the vocoder. The HiFi-GAN vocoder and the
data-dependent autoregressive generation loop are intentionally out of scope.

The forward pass is compiled with ``backend="tt"`` and its output is checked for
correlation against a CPU reference run, mirroring ``compiler_options.py``.
"""

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader, ModelVariant


def _build_model_and_inputs(dtype=torch.bfloat16):
    """Build the XTTS-v2 GPT core and its precomputed inputs via the loader."""
    loader = ModelLoader(ModelVariant.V2)
    model = loader.load_model(dtype_override=dtype).eval()
    inputs = loader.load_inputs(dtype_override=dtype)
    return loader, model, inputs


def run_xtts_v2(dtype=torch.bfloat16):
    """Run the XTTS-v2 GPT transformer core forward pass on a TT device."""
    _, model, inputs = _build_model_and_inputs(dtype)

    device = torch_xla.device()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Compile the transformer-core forward pass for the TT backend.
    model.compile(backend="tt")

    with torch.no_grad():
        output = model(**inputs)

    return output


def run_xtts_v2_cpu(dtype=torch.bfloat16):
    """Run the same forward pass on CPU for a correlation reference."""
    _, model, inputs = _build_model_and_inputs(dtype)

    with torch.no_grad():
        output = model(**inputs)

    return output


def post_process_output(output):
    """Print the speaker/text scenario and the predicted mel codes."""
    logits = output.cpu().float()

    # Sequence layout is [cond_latents | text_emb | mel_emb]; the mel-token
    # predictions for the audio frames are the final MEL_LEN positions.
    mel_len = ModelLoader.MEL_LEN
    mel_logits = logits[:, -mel_len:, :]
    predicted_codes = mel_logits.argmax(dim=-1)[0].tolist()

    print("XTTS-v2 GPT transformer-core forward pass")
    print(f"  Speaker:           {ModelLoader.SPEAKER}")
    print(f'  Text:              "{ModelLoader.SAMPLE_TEXT}"')
    print(f"  Output logits:     {tuple(logits.shape)}  (batch, seq, mel_vocab)")
    print(f"  Mel-code vocab:    {logits.shape[-1]}")
    print(f"  Predicted mel codes (first 16): {predicted_codes[:16]}")
    print(
        "  (these discrete mel codes are what the HiFi-GAN vocoder turns into a waveform)"
    )


def test_xtts_v2():
    """Check the TT forward pass is finite and correlates with the CPU reference."""
    xr.set_device_type("TT")

    tt_output = run_xtts_v2()
    cpu_output = run_xtts_v2_cpu()

    tt_output_cpu = tt_output.cpu().float()
    cpu_output = cpu_output.float()

    assert torch.isfinite(tt_output_cpu).all(), "TT output contains non-finite values"
    assert (
        tt_output_cpu.shape == cpu_output.shape
    ), f"shape mismatch: {tt_output_cpu.shape} vs {cpu_output.shape}"

    tt_flat = tt_output_cpu.flatten()
    cpu_flat = cpu_output.flatten()
    pcc = torch.corrcoef(torch.stack([tt_flat, cpu_flat]))[0, 1].item()

    print(f"PCC: {pcc}")
    print(f"Max diff: {(tt_output_cpu - cpu_output).abs().max()}")

    assert pcc > 0.99, f"PCC too low: {pcc}, expected > 0.99"


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output = run_xtts_v2()
    post_process_output(output)
