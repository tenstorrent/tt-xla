# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 — logic-equivalence test (CPU only).

The tt-xla bringup of XTTS-v2 does not run the model's native ``Xtts.inference``
(its autoregressive ``gpt.generate`` sampling loop is not traced). Instead the
loader precomputes ``gpt_codes`` and a custom monkey-patched ``forward`` runs the
deterministic tail of inference (a single GPT forward to produce latents followed
by the HiFiGAN decoder).

This test proves that custom decomposition is mathematically exact: it runs the
original ``Xtts.inference`` (greedy / ``do_sample=False`` so it is deterministic),
captures the exact ``gpt_codes`` produced by the internal generate step, feeds
those same codes into the custom ``forward``, and asserts the two output
waveforms have PCC ~= 1.0.

A PCC of ~1.0 here means any PCC gap observed on TT hardware is attributable to
device numerical precision, not to a flaw in the bringup decomposition.

This runs entirely on CPU and requires the optional ``coqui-tts`` dependency.
"""

import importlib.util

import pytest
import torch

# Optional heavy dependencies; skip cleanly where they are absent. Check for the
# TTS package without importing it: its import chain needs the isin_mps_friendly
# shim that ModelLoader.load_model installs (importing TTS here would fail).
pytest.importorskip("torchaudio")
if importlib.util.find_spec("TTS") is None:
    pytest.skip("coqui-tts (TTS) not installed", allow_module_level=True)

from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader, ModelVariant


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation coefficient between two tensors (flattened)."""
    a = a.flatten().to(torch.float64)
    b = b.flatten().to(torch.float64)
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


@pytest.mark.model_test
def test_custom_forward_matches_native_inference():
    torch.manual_seed(0)

    loader = ModelLoader(variant=ModelVariant.XTTS_V2)
    model = loader.load_model()  # float32, CPU; applies the monkey patches
    model.eval()

    text = loader.DEFAULT_TEXT
    language = loader.DEFAULT_LANGUAGE
    speaker = model.speaker_manager.speakers[loader.DEFAULT_SPEAKER]
    gpt_cond_latent = speaker["gpt_cond_latent"]
    speaker_embedding = speaker["speaker_embedding"]

    # --- 1. Native inference (deterministic), capturing the exact gpt_codes ---
    captured = {}
    original_generate = model.gpt.generate

    def capturing_generate(*args, **kwargs):
        codes = original_generate(*args, **kwargs)
        captured["gpt_codes"] = codes
        return codes

    model.gpt.generate = capturing_generate
    try:
        result = model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            do_sample=False,
            num_beams=1,
            temperature=0.75,
            top_k=50,
            top_p=0.85,
            length_penalty=1.0,
            repetition_penalty=10.0,
        )
    finally:
        model.gpt.generate = original_generate

    wav_native = torch.as_tensor(result["wav"])
    gpt_codes = captured["gpt_codes"]
    assert gpt_codes is not None, "Native inference did not invoke gpt.generate"

    # --- 2. Custom forward, fed the SAME gpt_codes produced above ---
    text_tokens = torch.IntTensor(
        model.tokenizer.encode(text.strip().lower(), lang=language)
    ).unsqueeze(0)
    text_len = torch.tensor([text_tokens.shape[-1]])
    expected_output_len = torch.tensor(
        [gpt_codes.shape[-1] * model.gpt.code_stride_len]
    )

    with torch.no_grad():
        wav_custom = model(
            text_tokens=text_tokens,
            text_len=text_len,
            gpt_codes=gpt_codes,
            expected_output_len=expected_output_len,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )

    # --- 3. The custom forward must reproduce native inference exactly ---
    pcc = _pcc(wav_native, wav_custom)
    print(f"native vs custom forward PCC: {pcc:.8f}")
    assert pcc >= 0.9999, f"Custom forward diverges from native inference: PCC={pcc}"
