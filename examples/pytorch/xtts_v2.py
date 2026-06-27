# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Coqui XTTS-v2 text-to-speech example.

Runs the compute-dominant core of XTTS-v2 — the autoregressive GPT-2 transformer
(378.9M params) plus final LayerNorm and mel-logits head — through a single
forward pass on a Tenstorrent device via ``torch.compile(backend="tt")``.

The input is the precomputed ``inputs_embeds`` sequence XTTS feeds to its GPT
core: ``cat([speaker conditioning latent, text embeddings, mel embeddings])``.
The forward produces logits over the discrete mel/audio codebook; the
``argmax`` of those logits is the sequence of audio codes that XTTS' HiFi-GAN
vocoder would decode into a waveform. The vocoder and the data-dependent
autoregressive sampling loop are out of device scope (unsupported ops / dynamic
output length), so this example demonstrates the transformer core that does the
heavy lifting, exactly as the tt-forge-models loader brings it up.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader, ModelVariant


# --------------------------------
# Model + inputs (via the loader)
# --------------------------------
def _load_core_and_inputs():
    """Build the XTTS-v2 GPT core and its precomputed embedding inputs in bf16."""
    loader = ModelLoader(ModelVariant.V2)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)
    return model, inputs


def xtts_v2():
    """Run the XTTS-v2 GPT core forward pass on a TT device."""
    device = torch_xla.device()

    model, inputs = _load_core_and_inputs()
    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled_model(**inputs)

    return output


# --------------------------------
# Human-readable result
# --------------------------------
def post_process_output(output):
    """Print the predicted audio-code sequence the GPT core emits.

    The forward returns mel logits of shape ``[batch, seq, num_audio_tokens]``;
    the per-position ``argmax`` is the discrete audio code XTTS' vocoder would
    decode into speech.
    """
    logits = output.cpu().float()
    predicted_codes = logits.argmax(dim=-1)[0]

    print(f"Mel logits shape: {tuple(logits.shape)}")
    print(f"Audio codebook size: {logits.shape[-1]}")
    print(f"Predicted audio-code sequence ({predicted_codes.numel()} codes):")
    print(predicted_codes.tolist())
    return predicted_codes


# --------------------------------
# Correctness guard for tt-xla CI
# --------------------------------
def test_xtts_v2():
    """Check the TT forward is finite, correctly shaped, and tracks CPU (PCC)."""
    xr.set_device_type("TT")

    model, inputs = _load_core_and_inputs()

    # CPU reference before moving the model to device.
    with torch.no_grad():
        cpu_output = model(**inputs)

    device = torch_xla.device()
    model = model.to(device)
    device_inputs = {k: v.to(device) for k, v in inputs.items()}
    compiled_model = torch.compile(model, backend="tt")
    with torch.no_grad():
        tt_output = compiled_model(**device_inputs)

    tt_cpu = tt_output.cpu().float()
    ref = cpu_output.float()

    assert torch.isfinite(tt_cpu).all(), "TT output contains non-finite values"
    assert tt_cpu.shape == ref.shape, f"shape {tt_cpu.shape} != {ref.shape}"

    pcc = torch.corrcoef(torch.stack([tt_cpu.flatten(), ref.flatten()]))[0, 1].item()
    print(f"PCC vs CPU: {pcc:.4f}")
    assert pcc > 0.95, f"PCC too low: {pcc}, expected > 0.95"

    print("XTTS-v2 GPT core forward matches CPU reference.")


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    output = xtts_v2()
    post_process_output(output)
