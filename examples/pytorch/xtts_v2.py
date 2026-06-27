# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
XTTS-v2 GPT transformer-core inference example.

XTTS-v2 (coqui/XTTS-v2) is a multilingual zero-shot voice-cloning TTS pipeline.
Its compute-dominant part is a 378.9M-parameter GPT-2 transformer core that
predicts discrete mel/audio tokens; the autoregressive sampling loop and the
HiFi-GAN vocoder around it are out of scope for a single static forward graph.

This example drives exactly that GPT core through the tt-forge-models loader. The
loader assembles ``inputs_embeds = cat([cond_latents, text_emb, mel_emb])`` from a
precomputed per-speaker conditioning latent and a fixed text/mel token sequence
(no reference audio needed), runs the core under ``torch.compile(backend="tt")``
on a Tenstorrent device, and produces the mel-token logits. The predicted mel
tokens are what the full pipeline would hand to the vocoder to synthesize speech.
"""

import torch
import torch_xla
import torch_xla.runtime as xr

from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader, ModelVariant

# The GPT core is brought up in bf16, matching the bringup/perf baseline.
DTYPE = torch.bfloat16


def _load_core(dtype=DTYPE):
    """Build the XTTS-v2 GPT core and its inputs on CPU via the loader.

    Returns the loader (it carries the demo text/speaker), the model and the
    ``inputs_embeds`` dict. ``load_model`` must run before ``load_inputs`` so the
    loader can record the mel offset on the model.
    """
    loader = ModelLoader(ModelVariant.V2)
    model = loader.load_model(dtype_override=dtype).eval()
    inputs = loader.load_inputs(dtype_override=dtype)
    return loader, model, inputs


def run_xtts_gpt_core_tt():
    """Run the XTTS-v2 GPT core on a TT device with torch.compile(backend='tt')."""
    device = torch_xla.device()

    loader, model, inputs = _load_core()

    model = model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    compiled = torch.compile(model, backend="tt")
    with torch.no_grad():
        logits = compiled(**inputs)

    return loader, logits


def run_xtts_gpt_core_cpu():
    """Run the XTTS-v2 GPT core on CPU for a correctness reference."""
    _, model, inputs = _load_core()
    with torch.no_grad():
        return model(**inputs)


def post_process_output(loader, logits):
    """Print the mel-token logits and the predicted discrete audio tokens."""
    logits = logits.cpu().float()
    pred_tokens = logits.argmax(dim=-1)[0].tolist()

    print(f"Speaker: {loader.DEFAULT_SPEAKER}")
    print(f'Text: "{loader.DEFAULT_TEXT}"')
    print(f"Mel-token logits shape: {tuple(logits.shape)}")
    print(f"Predicted mel/audio token IDs ({len(pred_tokens)} positions):")
    print(pred_tokens)


def test_xtts_v2():
    """Validate the TT GPT-core output against a CPU reference."""
    xr.set_device_type("TT")

    loader, tt_logits = run_xtts_gpt_core_tt()
    cpu_logits = run_xtts_gpt_core_cpu()

    tt_logits = tt_logits.cpu().float()
    cpu_logits = cpu_logits.float()

    # Output must be finite and keep the mel-region shape [B, mel_len, vocab].
    assert torch.isfinite(tt_logits).all(), "TT logits contain non-finite values"
    assert (
        tt_logits.shape == cpu_logits.shape
    ), f"shape mismatch: TT {tt_logits.shape} vs CPU {cpu_logits.shape}"

    tt_flat = tt_logits.flatten()
    cpu_flat = cpu_logits.flatten()
    pcc = torch.corrcoef(torch.stack([tt_flat, cpu_flat]))[0, 1].item()

    print(f"PCC: {pcc}")
    print(f"Max diff: {(tt_logits - cpu_logits).abs().max().item()}")

    # bf16 reproduces the CPU reference closely (a few token-boundary ties aside).
    assert pcc > 0.98, f"PCC too low: {pcc}, expected > 0.98"
    print("XTTS-v2 GPT core matches the CPU reference.")


if __name__ == "__main__":
    xr.set_device_type("TT")

    loader, logits = run_xtts_gpt_core_tt()
    post_process_output(loader, logits)
