# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
XTTS-v2 text-to-speech (GPT-core) example on Tenstorrent hardware.

coqui/XTTS-v2 is a Coqui TTS pipeline whose compute-dominant component is an
autoregressive GPT-2 backbone ("the GPT core"). It consumes a sequence of
speaker-conditioning latents + text-token embeddings + mel-token embeddings and
predicts the next mel (audio-codebook) token; the downstream HiFi-GAN vocoder
that turns mel codes into a waveform is out of scope for this single-forward
bringup. This example drives the GPT core through the tt-forge-models loader,
compiles it for the TT backend, and compares the mel-token logits against a CPU
reference -- mirroring how the core is exercised inside the XTTS inference loop.

Modelled after compiler_options.py: a single bf16 forward, TT-vs-CPU PCC.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader, ModelVariant

# Run the GPT core in bf16 -- the bringup/perf baseline for this model. fp32 +
# bare eager execution does not route to the TT backend; bf16 + a tt-backend
# compile reproduces the bringup PCC.
DTYPE = torch.bfloat16


def _build_loader():
    return ModelLoader(ModelVariant.XTTS_V2)


def run_xtts_gpt_core_on_tt():
    """Run the XTTS-v2 GPT core on the TT device and return its mel logits."""
    device = torch_xla.device()

    loader = _build_loader()
    # load_model before load_inputs: the loader populates its conditioning prior
    # on first checkpoint access, and both calls share that cached state.
    model = loader.load_model(dtype_override=DTYPE).eval()
    inputs = loader.load_inputs(dtype_override=DTYPE)

    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    compiled = torch.compile(model, backend="tt")
    with torch.no_grad():
        mel_logits = compiled(**inputs)
    return mel_logits


def run_xtts_gpt_core_on_cpu():
    """Run the same GPT core on CPU for a reference comparison."""
    loader = _build_loader()
    model = loader.load_model(dtype_override=DTYPE).eval()
    inputs = loader.load_inputs(dtype_override=DTYPE)
    with torch.no_grad():
        return model(**inputs)


def post_process_output(mel_logits):
    """Print a human-readable result: the predicted mel (audio-codebook) tokens."""
    mel_logits = mel_logits.cpu().float()
    pred_tokens = mel_logits.argmax(dim=-1)[0].tolist()
    print(f"Mel-token logits shape: {tuple(mel_logits.shape)}")
    print(f"Predicted mel/audio token ids ({len(pred_tokens)}): {pred_tokens}")
    return pred_tokens


def test_xtts_v2():
    """Guard the example: finite output, expected shape, and TT-vs-CPU agreement."""
    xr.set_device_type("TT")

    tt_logits = run_xtts_gpt_core_on_tt().cpu().float()
    cpu_logits = run_xtts_gpt_core_on_cpu().float()

    # Expected static shape: (batch=1, MEL_SEQ_LEN, num_audio_tokens=1026).
    expected_shape = (1, ModelLoader.MEL_SEQ_LEN, 1026)
    assert tt_logits.shape == expected_shape, (
        f"Unexpected output shape: {tuple(tt_logits.shape)} != {expected_shape}"
    )
    assert torch.isfinite(tt_logits).all(), "Non-finite values in TT output"

    pcc = torch.corrcoef(
        torch.stack([tt_logits.flatten(), cpu_logits.flatten()])
    )[0, 1].item()
    print(f"PCC: {pcc}")
    print(f"Max diff: {(tt_logits - cpu_logits).abs().max().item()}")

    assert pcc > 0.98, f"PCC too low: {pcc}, expected > 0.98"


if __name__ == "__main__":
    xr.set_device_type("TT")

    mel_logits = run_xtts_gpt_core_on_tt()
    print("Success! XTTS-v2 GPT core ran on the TT device.")
    post_process_output(mel_logits)
