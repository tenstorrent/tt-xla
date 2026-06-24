# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lumina-Image-2.0 — nightly e2e pipeline test with per-stage PCC checks.

Lumina-Image-2.0 is a flow-matching text-to-image diffusion model. This drives
the shared ``LuminaImagePipeline`` from ``tt_forge_models`` (the same pipeline the
image-gen benchmark uses) end-to-end, gates its numerics per TT stage and asserts
the saved image dimensions.

All three learned components run on TT (``pipeline.TT_COMPONENTS``),
tensor-parallel sharded on the ("batch", "model") mesh: the Gemma-2 text encoder
(once per CFG branch), the DiT (2 forwards per denoising step for CFG) and the
AutoencoderKL decoder (once). Tokenizer, scheduler, latent sampling and the CFG
combine stay on CPU.
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
from third_party.tt_forge_models.lumina_image.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.lumina_image.pytorch.src.pipeline import (
    HEIGHT,
    PROMPT,
    SEED,
    VAE_OPT_LEVEL,
    WIDTH,
    LuminaImageConfig,
    LuminaImagePipeline,
    _Gemma2PenultimateEncoder,
)

VARIANT_NAME = ModelVariant.TRANSFORMER
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

OUTPUT_PATH = "lumina_image_2_output.png"
NUM_INFERENCE_STEPS = 30
PCC_THRESHOLD = 0.99
PCC_GOLDEN_DTYPE = torch.bfloat16
# Pipeline attribute, display name and CPU-twin builder for each gateable
# component, in pipeline order. Keyed by the name ``TT_COMPONENTS`` uses, so the
# gate can ask the pipeline which of these are actually on TT rather than
# duplicating that decision here.
#
# The Gemma-2 twin is re-wrapped in ``_Gemma2PenultimateEncoder`` for the same
# reason the pipeline wraps the device-side encoder: Lumina2 conditions on
# ``hidden_states[-2]``, which the loader's own wrapper does not return.
_GATED_COMPONENTS = {
    "text_encoder": (
        "text_encoder (Gemma-2)",
        lambda: _Gemma2PenultimateEncoder(
            ModelLoader(ModelVariant.TEXT_ENCODER)
            .load_model(dtype_override=PCC_GOLDEN_DTYPE)
            .encoder
        ),
    ),
    "transformer": (
        "transformer (DiT)",
        lambda: ModelLoader(ModelVariant.TRANSFORMER).load_model(
            dtype_override=PCC_GOLDEN_DTYPE
        ),
    ),
    "vae": (
        "vae (AutoencoderKL decoder)",
        lambda: ModelLoader(ModelVariant.VAE).load_model(
            dtype_override=PCC_GOLDEN_DTYPE
        ),
    ),
}


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    # Both sides upcast to fp32 so the correlation itself is computed cleanly,
    # whatever precision the two forwards ran at.
    return float(
        _PCC_EVALUATOR._compare_pcc(
            device_out.to("cpu").float(), golden_out.to("cpu").float(), _PCC_CONFIG
        )
    )


def _to_golden(x):
    """Move to CPU at ``PCC_GOLDEN_DTYPE`` so the twin sees what TT consumed.

    Integer/bool tensors (input ids, attention masks) are left untouched.
    """
    if not isinstance(x, torch.Tensor):
        return x
    x = x.to("cpu")
    return x.to(PCC_GOLDEN_DTYPE) if x.is_floating_point() else x


def _attach_pcc_checks(pipeline: LuminaImagePipeline) -> None:
    """Wrap every TT component's forward with a CPU-twin PCC check, asserted per
    forward. The pipeline keeps using the real TT output.

    Which components get gated is read off the pipeline (``_tt_resident()``,
    i.e. ``TT_COMPONENTS``) rather than hardcoded here, so the gate follows the
    device map instead of drifting from it. A component left on CPU is skipped:
    gating it would compare CPU against CPU -- PCC 1.0 by construction, paid for
    with a full host copy of the weights.

    Must be called after ``setup()``, which is what shards the components, moves
    them to the XLA device and wraps them for the ``tt`` backend -- the objects
    wrapped here are exactly the ones ``generate()`` invokes. Each twin is loaded
    lazily on that component's first forward, so an unused component costs
    nothing.
    """

    def attach(module, name, build_twin, pick=lambda out: out):
        orig_forward = module.forward
        twin = {"model": None}
        step = {"n": 0}

        def _cpu_twin():
            if twin["model"] is None:
                logger.info("[PCC] loading CPU {} twin: {}", PCC_GOLDEN_DTYPE, name)
                twin["model"] = build_twin()
            return twin["model"]

        # The pipeline calls every component positionally, so the twin gets the
        # same args on CPU; a future kwargs call raises rather than mismatching.
        # ``_to_golden`` rather than a plain ``.to("cpu")``: the DiT takes an fp32
        # timestep alongside its bf16 activations, and a bf16 twin fed fp32 would
        # raise a dtype mismatch in the timestep embedder.
        def wrapped_forward(*args):
            out = orig_forward(*args)
            golden = pick(_cpu_twin()(*[_to_golden(a) for a in args]))

            step["n"] += 1
            pcc = _pcc(pick(out), golden)
            logger.info("[PCC] {} forward {}: pcc={:.6f}", name, step["n"], pcc)
            assert (
                pcc >= PCC_THRESHOLD
            ), f"{name} forward {step['n']} PCC {pcc:.6f} below {PCC_THRESHOLD}"
            return out

        module.forward = wrapped_forward

    on_tt = pipeline._tt_resident()
    for component, (name, build_twin) in _GATED_COMPONENTS.items():
        if component not in on_tt:
            logger.info("[PCC] {} runs on CPU — not gated", component)
            continue
        attach(getattr(pipeline, component), name, build_twin)


def _scope_vae_to_opt_level(pipeline: LuminaImagePipeline) -> None:
    """Compile the VAE decode graph at ``VAE_OPT_LEVEL`` (opt-level 1).

    Wraps ``vae.forward`` the same way ``_attach_pcc_checks`` does, so call this
    after ``setup()`` (which is what puts the compiled module in place) and
    before the PCC gate, so the gate's own wrapper nests outside this one and its
    CPU twin stays out of the scope.
    """
    vae = pipeline.vae
    orig_vae_forward = vae.forward

    def vae_at_opt_level(z):
        with pipeline._vae_compile_options():
            return orig_vae_forward(z).to("cpu")

    vae.forward = vae_at_opt_level
    logger.info(f"[COMPILE] VAE decode scoped to optimization_level={VAE_OPT_LEVEL}")


def run_lumina_image_pipeline(
    output_path: str = OUTPUT_PATH,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    shard: bool = True,
    check_pcc: bool = True,
):
    """Run the Lumina-Image-2.0 pipeline (all components on TT) and save image.

    Runs the pipeline's own defaults throughout — including ``cfg_trunc_ratio``
    (1.0, CFG on every step) and ``pad_negative_caption`` — with one deviation:
    the VAE decode is compiled at ``VAE_OPT_LEVEL``, see
    ``_scope_vae_to_opt_level``. Pass ``check_pcc=False`` for a throughput-only
    run: the gate replays every TT forward on CPU, which dominates wall-clock.
    """
    pipeline = LuminaImagePipeline(
        config=LuminaImageConfig(
            on_tt=True,
            shard=shard,
            num_inference_steps=num_inference_steps,
        )
    )
    pipeline.setup()

    # Before the PCC gate: the gate wraps vae.forward too, and this one has to be
    # the inner wrapper so the decode is lowered under the opt-level override.
    _scope_vae_to_opt_level(pipeline)

    if check_pcc:
        _attach_pcc_checks(pipeline)

    # ``generate`` post-processes via the diffusers image processor and returns a
    # list of PIL images (output_type="pil").
    images = pipeline.generate(prompt=PROMPT, seed=SEED)
    images[0].save(output_path)
    return output_path


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.qb2_blackhole
@pytest.mark.tensor_parallel
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_lumina_image_2_pipeline():
    """Run the full Lumina-Image-2.0 pipeline end to end with per-stage PCC.

    Gemma-2 text encoder, DiT and AutoencoderKL decoder sharded on TT; tokenizer,
    scheduler, latent sampling and the CFG combine on CPU.

    Runs the pipeline's own defaults, which include ``pad_negative_caption`` --
    the workaround that keeps both CFG forwards on one DiT executable. That is
    load-bearing for this test: without it the DiT decays to 0.84 by step 4 and
    never reaches the VAE decode (see
    ``LuminaImageConfig.pad_negative_caption``).
    """
    xr.set_device_type("TT")

    output_file = Path(OUTPUT_PATH)
    if output_file.exists():
        output_file.unlink()

    run_lumina_image_pipeline(
        output_path=OUTPUT_PATH,
        num_inference_steps=NUM_INFERENCE_STEPS,
        shard=True,
    )

    assert output_file.exists(), f"Output image {OUTPUT_PATH} was not created"

    with Image.open(OUTPUT_PATH) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"

    logger.info(f"Output image saved to {OUTPUT_PATH} ({width}x{height})")
