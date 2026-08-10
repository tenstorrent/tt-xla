# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-Image — nightly e2e text-to-image pipeline test with per-step PCC checks.

GLM-Image is a diffusion text-to-image model whose DiT transformer runs
tensor-parallel across a multi-chip mesh, while the AR vision-language
encoder, the T5 glyph text encoder, the FlowMatchEuler scheduler and the VAE
decode stay on CPU. This drives the shared GlmImagePipeline from
tt_forge_model` (the same pipeline the image-gen benchmark uses) end-to-end,
gates its numerics per denoising step and asserts the saved image dimensions.

The test fails fast the moment any DiT step drops below ``PCC_THRESHOLD``. The pipeline
itself keeps using the TT outputs.

The DiT is the only component that runs on TT in this pipeline — the T5
encoder, the vision-language encoder and the VAE all run on CPU here.
"""

from pathlib import Path

import pytest
import torch
import torch_xla.runtime as xr
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from PIL import Image
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.glm_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.glm_image.pytorch.src.pipeline import (
    HEIGHT,
    PROMPT,
    WIDTH,
    GlmImageConfig,
    GlmImagePipeline,
)

VARIANT_NAME = ModelVariant.TRANSFORMER
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

SEED = 42
NUM_INFERENCE_STEPS = 30
# bf16 DiT on TT vs a clean fp32 CPU reference. Mirrors tt-xla#5480's 0.99 gate;
# tune against the first nightly if the bf16/fp32 gap sits below this.
PCC_THRESHOLD = 0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _attach_dit_pcc_check(pipeline: GlmImagePipeline) -> None:
    """Wrap the pipeline's DiT forward with an inline fp32-CPU-twin PCC check.

    Only the DiT transformer runs on TT, so it is the sole component gated here.
    After each TT forward the same inputs are replayed on a lazily loaded fp32
    CPU twin and PCC is asserted inline, so the test fails fast on the first
    diverging denoising step (checked per CFG forward: conditional + uncond).

    Must be called after ``pipeline.setup()`` — setup shards the DiT and moves
    it to the XLA device, and ``pipeline.transformer`` is the object
    ``generate()`` invokes.
    """
    transformer = pipeline.transformer
    # Bound method of the (already-patched) class forward; captured before the
    # instance attribute below shadows it.
    orig_forward = transformer.forward

    twin = {"model": None}
    step = {"n": 0}

    def _cpu_twin():
        # Loaded on first use: a fresh fp32 CPU copy of the DiT. It stays plain
        # fp32 (no bf16 cast, no force_fp32_layernorm patch) so it is a clean
        # golden reference for the bf16 TT DiT.
        if twin["model"] is None:
            logger.info("[PCC] loading CPU fp32 DiT twin")
            twin["model"] = ModelLoader(ModelVariant.TRANSFORMER).load_model(
                dtype_override=torch.float32
            )
        return twin["model"]

    def _cpu(x):
        # Move to CPU and upcast floats to fp32 so the twin sees the same values
        # the TT DiT consumed; leave int/bool tensors (prior ids/drop) untouched.
        if not isinstance(x, torch.Tensor):
            return x
        x = x.to("cpu")
        return x.float() if x.is_floating_point() else x

    def wrapped_forward(*args, **kwargs):
        # Real TT forward — the pipeline continues with this output.
        out = orig_forward(*args, **kwargs)
        device_sample = out[0] if isinstance(out, (tuple, list)) else out
        device_sample = device_sample.to("cpu").float()

        # Replay the same inputs on the fp32 CPU twin. The pipeline always calls
        # the DiT with these keywords (see GlmImagePipeline.generate._dit).
        golden_sample = _cpu_twin()(
            _cpu(kwargs["hidden_states"]),
            _cpu(kwargs["encoder_hidden_states"]),
            _cpu(kwargs["prior_token_id"]),
            _cpu(kwargs["prior_token_drop"]),
            _cpu(kwargs["timestep"]),
            _cpu(kwargs["target_size"]),
            _cpu(kwargs["crop_coords"]),
        )

        step["n"] += 1
        pcc = _pcc(device_sample, golden_sample)
        logger.info(f"[PCC] dit forward {step['n']}: pcc={pcc:.6f}")
        assert (
            pcc >= PCC_THRESHOLD
        ), f"DiT forward {step['n']} PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"

        return out

    transformer.forward = wrapped_forward


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.lb_blackhole
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_pipeline():
    """Run the GLM-Image pipeline (DiT tensor-parallel) with per-step DiT PCC."""
    xr.set_device_type("TT")

    pipeline = GlmImagePipeline(
        config=GlmImageConfig(num_inference_steps=NUM_INFERENCE_STEPS)
    )
    pipeline.setup()

    # Gate the DiT (the only TT component) against an fp32 CPU twin per step.
    _attach_dit_pcc_check(pipeline)

    # ``generate`` post-processes via the diffusers image processor and returns a
    # list of PIL images (output_type="pil").
    images = pipeline.generate(prompt=PROMPT, seed=SEED)

    output_path = "glm_image_pipeline_output.png"
    images[0].save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
