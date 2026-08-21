# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
LTX-2.3 — video VAE (CausalVideoAutoencoder) single-device component tests.

The video VAE is ported from the native ``ltx_core`` codebase
(github.com/Lightricks/LTX-2). Both halves are built from the checkpoint's
embedded ``vae`` config and loaded with the REAL weights copied out of the
cached Pro/dev checkpoint (``vae.{decoder,encoder}.*`` + per-channel stats):

    decoder  407.2M params   latent (1,128,2,8,8) -> video (1,3,9,256,256)
    encoder  318.9M params   video (1,3,9,256,256) -> latent (1,128,2,8,8)

Compression is 8x temporal / 32x spatial; the two shapes are exact round-trip
inverses. Both fit a single TT chip. Each test runs one forward and compares
CPU vs TT output at PCC 0.99.
"""

import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig

from tests.infra.testers.compiler_config import CompilerConfig

from third_party.tt_forge_models.ltx2_3.pytorch import ModelLoader, ModelVariant


def test_ltx2_3_video_vae_decoder():
    _run(ModelVariant.VIDEO_VAE_DECODER)


def test_ltx2_3_video_vae_encoder():
    _run(ModelVariant.VIDEO_VAE_ENCODER)


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
