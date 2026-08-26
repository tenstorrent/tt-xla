# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 — audio VAE + vocoder single-device component tests.

Ported from the native ``ltx_core`` codebase (github.com/Lightricks/LTX-2), built
from the checkpoint's embedded configs and loaded with REAL weights from the cached
Pro/dev checkpoint:

    audio decoder  31.9M   latent (1,8,64,16)  -> spectrogram (1,2,253,64)
    audio encoder  21.3M   spectrogram (1,2,256,64) -> latent (1,8,64,16)
    vocoder       128.5M   mel (1,2,64,64)     -> waveform (1,2,30720)

The audio VAE halves are stereo mel conv-autoencoders (ch_mult=[1,2,4], z=8). The
vocoder is a BigVGAN-v2 + band-width-extension stack (VocoderWithBWE) that runs its
forward in fp32 and contains an internal STFT — the highest-risk of the three on
device. Each test runs one forward and compares CPU vs TT at PCC 0.99.
"""

import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.ltx2_3.pytorch import ModelLoader, ModelVariant


def test_ltx2_3_audio_vae_decoder():
    _run(ModelVariant.AUDIO_VAE_DECODER)


def test_ltx2_3_audio_vae_encoder():
    _run(ModelVariant.AUDIO_VAE_ENCODER)


def test_ltx2_3_vocoder():
    _run(ModelVariant.VOCODER)


def _run(variant: ModelVariant):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    loader = ModelLoader(variant)
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
