# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Pyramid Flow (miniFLUX) - nightly e2e pipeline test. WORK IN PROGRESS.

Runs the full text-to-video pipeline end-to-end: the CLIP text encoder, the
1.97B ``PyramidFluxTransformer`` DiT and the ``CausalVideoVAE`` decoder all run
on Tenstorrent via ``torch.compile(backend="tt")``. The pyramid flow-matching
sampler stays on CPU - it is control flow, not a net.

Per-component bringup that this test assembles:
  - DiT + CLIP text encoder: tt-forge-models#841
  - CausalVideoVAE decode:   tt-forge-models#900 (PCC 0.999956, T=1, n150)

The reusable pipeline implementation belongs in ``tt_forge_models`` next to the
components it drives, the way SD1.5's does
(``third_party/tt_forge_models/stable_diffusion_1_5/pytorch/pipeline.py``), so
this file only wires it into CI. Until that companion PR lands the import below
fails and the test skips rather than erroring at collection.

Scope, for now: one temporal unit (``temp=1``, a single latent frame), 384p,
single device. That is the shape #900 validated the VAE decode at; the
multi-frame chunked decode still does not trace.
"""

from pathlib import Path

import pytest
import torch_xla.runtime as xr
from infra import RunMode
from loguru import logger
from utils import BringupStatus, Category, ModelGroup

try:
    from third_party.tt_forge_models.pyramid_flow.pytorch.pipeline import (
        PyramidFlowConfig,
        PyramidFlowPipeline,
        save_video,
    )

    PIPELINE_AVAILABLE = True
except ImportError:
    # The pipeline module lands with the companion tt-forge-models PR; skip
    # instead of failing collection for every other test in the file's package.
    PIPELINE_AVAILABLE = False

PROMPT = (
    "A red double-decker bus driving along a sunny coastal road, "
    "waves breaking on the beach below"
)
NEGATIVE_PROMPT = ""
SEED = 12345
VARIANT = "diffusion_transformer_384p"
# 384p renders 640x384. temp=1 -> a single frame (temp=k -> 8k-7 frames).
WIDTH = 640
HEIGHT = 384
TEMP = 1
GUIDANCE_SCALE = 7.0
FPS = 24


def run_pyramid_flow_pipeline(
    output_path: str = "pyramid_flow_output.mp4",
    num_inference_steps: tuple = (8, 8, 8),
):
    """Run the Pyramid Flow pipeline with CLIP, the DiT and the VAE on TT."""
    config = PyramidFlowConfig(
        variant=VARIANT,
        text_encoder_on_tt=True,
        transformer_on_tt=True,
        vae_on_tt=True,
    )
    pipeline = PyramidFlowPipeline(config=config)
    pipeline.setup()

    frames = pipeline.generate(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        temp=TEMP,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=list(num_inference_steps),
        seed=SEED,
    )

    save_video(frames, output_path, fps=FPS)
    return output_path


@pytest.mark.skipif(
    not PIPELINE_AVAILABLE,
    reason="Pyramid Flow pipeline module lands with the companion tt-forge-models PR",
)
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="PyramidFlow_Pipeline",
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.NOT_STARTED,
)
def test_pyramid_flow_pipeline():
    """Run the full Pyramid Flow text-to-video pipeline with the DiT on TT."""
    xr.set_device_type("TT")

    output_path = "pyramid_flow_output.mp4"
    output_file = Path(output_path)
    if output_file.exists():
        output_file.unlink()

    run_pyramid_flow_pipeline(output_path=output_path)

    assert output_file.exists(), f"Output video {output_path} was not created"
    logger.info(f"Output video saved to {output_path} ({WIDTH}x{HEIGHT}, temp={TEMP})")
