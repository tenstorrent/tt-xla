# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""GLM-Image — nightly e2e text-to-image pipeline test with per-step PCC checks.

GLM-Image is a diffusion text-to-image model whose T5 glyph text encoder, DiT
transformer and VAE decoder all run tensor-parallel across a multi-chip mesh,
while the AR vision-language encoder and the FlowMatchEuler scheduler stay on
CPU. This drives the shared GlmImagePipeline from
tt_forge_model` (the same pipeline the image-gen benchmark uses) end-to-end,
gates its numerics per denoising step and asserts the saved image dimensions.

Every component that runs on TT is gated against an fp32 CPU twin, inline, per
forward: the T5 glyph encode (twice -- prompt and empty negative prompt), each
DiT forward (two per denoising step under CFG) and the single VAE decode. The
test fails fast the moment one drops below its threshold; the pipeline itself
keeps using the TT outputs.

The AR vision-language encoder is not gated -- it stays on CPU, so there is
nothing to compare against.
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
# TT vs a clean fp32 CPU reference, per component. All three mirror tt-xla#5480's
# 0.99 gate; they are separate constants so one component can be tuned against
# the first nightly without loosening the others.
#
# The T5 encoder and DiT run bf16 on device. The VAE decoder runs fp32 (its
# config sets ``force_upcast: true``), so its gate is a like-for-like fp32
# comparison rather than a bf16-vs-fp32 one -- see VAE_DTYPE in the pipeline.
DIT_PCC_THRESHOLD = 0.99
TEXT_ENCODER_PCC_THRESHOLD = 0.99
VAE_PCC_THRESHOLD = 0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _cpu(x):
    """Move to CPU and upcast floats to fp32 for the twin.

    Int/bool tensors (prior ids/drop, input ids, attention masks) are left
    untouched, and non-tensors pass through.
    """
    if not isinstance(x, torch.Tensor):
        return x
    x = x.to("cpu")
    return x.float() if x.is_floating_point() else x


def _first(out):
    """First tensor of a forward's output (the DiT returns a 1-tuple)."""
    return out[0] if isinstance(out, (tuple, list)) else out


def _twin(variant: ModelVariant):
    """Load the fp32 CPU golden for a component.

    Plain fp32 (no bf16 cast, no force_fp32_layernorm patch) so it is a clean
    reference for the bf16 TT component.
    """
    return ModelLoader(variant).load_model(dtype_override=torch.float32)


def _attach_pcc_checks(pipeline: GlmImagePipeline) -> None:
    """Wrap every TT component's forward with an inline fp32-CPU-twin PCC check.

    After each TT forward the same inputs are replayed on a lazily loaded fp32
    CPU twin and PCC is asserted inline, so the test fails fast on the first
    divergence rather than at the end of a 30-step generation. The pipeline
    keeps using the real TT output either way.

    Must be called after ``pipeline.setup()`` — setup is what shards each
    component, moves it to the XLA device and installs the compiled forward that
    ``generate()`` ends up invoking.
    """

    def attach(module, name, threshold, build_twin, replay):
        # Bound method of the (already compiled / patched) forward; captured
        # before the instance attribute below shadows it.
        orig_forward = module.forward

        twin = {"model": None}
        step = {"n": 0}

        def cpu_twin():
            if twin["model"] is None:
                logger.info(f"[PCC] loading CPU fp32 twin: {name}")
                twin["model"] = build_twin()
            return twin["model"]

        def wrapped_forward(*args, **kwargs):
            # Real TT forward — the pipeline continues with this output.
            out = orig_forward(*args, **kwargs)
            device_sample = _first(out).to("cpu").float()
            golden_sample = _first(replay(cpu_twin(), *args, **kwargs)).float()

            step["n"] += 1
            pcc = _pcc(device_sample, golden_sample)
            logger.info(f"[PCC] {name} forward {step['n']}: pcc={pcc:.6f}")
            assert (
                pcc >= threshold
            ), f"{name} forward {step['n']} PCC {pcc:.6f} below threshold {threshold}"

            return out

        module.forward = wrapped_forward

    if pipeline.config.text_encoder_on_tt:
        # TTTextEncoder.wrapped is the compiled tensor-in/tensor-out T5 module;
        # the pipeline's CPU-side mask gather sits outside it. Called
        # positionally as (input_ids, attention_mask); the twin is the raw
        # T5EncoderModel, hence the .last_hidden_state pick.
        attach(
            pipeline.text_encoder.wrapped,
            "text_encoder (T5)",
            TEXT_ENCODER_PCC_THRESHOLD,
            lambda: _twin(ModelVariant.TEXT_ENCODER),
            lambda twin, input_ids, attention_mask: twin(
                input_ids=_cpu(input_ids),
                attention_mask=_cpu(attention_mask),
            ).last_hidden_state,
        )

    if pipeline.config.transformer_on_tt:
        # The pipeline always calls the DiT with these keywords (see
        # GlmImagePipeline.generate._dit); the twin wrapper takes them
        # positionally in this order.
        attach(
            pipeline.transformer,
            "transformer (DiT)",
            DIT_PCC_THRESHOLD,
            lambda: _twin(ModelVariant.TRANSFORMER),
            lambda twin, *args, **kwargs: twin(
                _cpu(kwargs["hidden_states"]),
                _cpu(kwargs["encoder_hidden_states"]),
                _cpu(kwargs["prior_token_id"]),
                _cpu(kwargs["prior_token_drop"]),
                _cpu(kwargs["timestep"]),
                _cpu(kwargs["target_size"]),
                _cpu(kwargs["crop_coords"]),
            ),
        )

    if pipeline.config.vae_on_tt:
        # pipeline.vae_decoder is a VAEDecoderWrapper called as (z); the twin is
        # the same wrapper around an fp32 CPU VAE.
        attach(
            pipeline.vae_decoder,
            "vae decoder",
            VAE_PCC_THRESHOLD,
            lambda: _twin(ModelVariant.VAE),
            lambda twin, z: twin(_cpu(z)),
        )


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
    """Run the GLM-Image pipeline (tensor-parallel) with per-forward PCC gates."""
    xr.set_device_type("TT")

    pipeline = GlmImagePipeline(
        config=GlmImageConfig(num_inference_steps=NUM_INFERENCE_STEPS)
    )
    pipeline.setup()

    # Gate every TT component (T5 encoder, DiT, VAE decoder) against an fp32 CPU
    # twin, per forward.
    _attach_pcc_checks(pipeline)

    # ``generate`` post-processes via the diffusers image processor and returns a
    # list of PIL images (output_type="pil").
    images = pipeline.generate(prompt=PROMPT, seed=SEED)

    output_path = "glm_image_pipeline_output_sep2.png"
    images[0].save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
