# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Lumina-Image-2.0 — nightly e2e pipeline test with per-stage PCC checks.

Lumina-Image-2.0 is a flow-matching text-to-image diffusion model. This drives
the shared ``LuminaImagePipeline`` from ``tt_forge_models`` (the same pipeline the
image-gen benchmark uses) end-to-end, gates its numerics per TT stage and asserts
the saved image dimensions.

Like GLM-Image, only the DiT runs on TT (``pipeline.TT_COMPONENTS``),
tensor-parallel sharded on the ("batch", "model") mesh — 2 forwards per denoising
step for CFG. The Gemma-2 text encoder and the AutoencoderKL decoder run on CPU:
neither is validated on TT today, and the VAE in particular produced a pure-noise
image when run there (see ``TT_COMPONENTS`` for the details).

The DiT is gated against a lazily loaded CPU twin. The test fails fast the moment
it drops below ``PCC_THRESHOLD``. The pipeline itself keeps using the TT outputs.

Two of the pipeline's defaults are workarounds this test depends on rather than
merely tolerates — ``compile_on_tt`` (composite ops, without which the DiT
measures 0.918) and ``pad_negative_caption`` (one DiT executable, without which it
decays to 0.84 by step 4). The open bug behind the second one, and the evidence
that isolated it to two alternating executables, is documented at
``LuminaImageConfig.pad_negative_caption``.

Each twin is a host copy of the same weights at ``PCC_GOLDEN_DTYPE``, so gating
costs host RAM per gated component and replays every TT forward on CPU — narrow
``PCC_COMPONENTS`` to trade coverage for runtime.
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
    WIDTH,
    LuminaImageConfig,
    LuminaImagePipeline,
)

VARIANT_NAME = ModelVariant.TRANSFORMER
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

OUTPUT_PATH = "lumina_image_2_output.png"
NUM_INFERENCE_STEPS = 30
# Mirrors tt-xla#5480's 0.99 gate. With ``pad_negative_caption`` on (the pipeline
# default) both CFG forwards share one DiT executable and every one of the 60
# forwards measures >=0.99979, so this has real headroom. Turning that workaround
# off drops the uncond forwards to 0.84 by step 4 -- the open two-executable bug
# described at ``LuminaImageConfig.pad_negative_caption``.
PCC_THRESHOLD = 0.99
# dtype of the CPU twins. bf16 keeps this an apples-to-apples TT-vs-CPU check at
# the *same* precision — the comparison the per-component graph tests make
# (``run_graph_test`` runs the identical bf16 model on CPU), so a failure here is
# a device/compiler divergence rather than bf16 quantization error. Set to
# ``torch.float32`` to measure total error against a clean reference instead;
# expect materially lower PCC, since the whole 26-block DiT's bf16 rounding then
# counts against the device.
PCC_GOLDEN_DTYPE = torch.bfloat16
# Which TT stages to gate against a CPU twin. Each entry costs one host copy of
# that component plus a CPU replay of every one of its forwards (the DiT is by
# far the most expensive: 2 replays per denoising step). Narrow this set — or
# empty it — for a throughput-only run.
#
# Only the DiT: it is the only component ``TT_COMPONENTS`` puts on TT. Gating the
# text encoder or VAE now would compare CPU against CPU — PCC 1.0 by construction,
# paid for with a full host copy of each. Widen this in step with
# ``TT_COMPONENTS``, not independently.
PCC_COMPONENTS = frozenset({"transformer"})


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


def _attach_pcc_checks(
    pipeline: LuminaImagePipeline,
    *,
    components=PCC_COMPONENTS,
    threshold: float = PCC_THRESHOLD,
) -> dict:
    """Gate every TT stage of the pipeline against a CPU twin, inline.

    Returns the ``{stage_name: [pcc, ...]}`` history in forward order. PCC is
    asserted inline, so that history only covers forwards up to a failure.

    After each TT forward the same inputs are replayed on a lazily loaded
    ``PCC_GOLDEN_DTYPE`` CPU twin of that component and PCC is asserted
    immediately, so the test fails on the first diverging stage rather than on
    the final image.

    Must be called after ``pipeline.setup()`` — setup shards the components and
    moves them to the XLA device, and the objects wrapped here
    (``pipeline.transformer``, ``pipeline.vae``, ``pipeline._encode_on_tt``) are
    exactly what ``generate()`` invokes.

    Note the DiT twin exercises the same host-side real-valued RoPE rewrite the
    TT model does (``load_transformer`` patches it globally, and it is bit-exact
    against diffusers' complex form — see ``verify_freqs.py``). This gate
    therefore measures TT-vs-CPU numerics, not the RoPE rewrite itself.
    """
    twins: dict = {}
    history: dict = {}

    def _twin(variant: ModelVariant):
        if variant not in twins:
            logger.info(f"[PCC] loading CPU {PCC_GOLDEN_DTYPE} twin: {variant}")
            twins[variant] = ModelLoader(variant).load_model(
                dtype_override=PCC_GOLDEN_DTYPE
            )
        return twins[variant]

    def _gate(name: str, device_out, golden_out) -> None:
        pcc = _pcc(device_out, golden_out)
        history.setdefault(name, []).append(pcc)
        n = len(history[name])
        logger.info(f"[PCC] {name} forward {n}: pcc={pcc:.6f}")
        assert (
            pcc >= threshold
        ), f"{name} forward {n} PCC {pcc:.6f} below threshold {threshold}"

    # ── Gemma-2 text encoder ───────────────────────────────────────────
    if "text_encoder" in components:
        orig_encode = pipeline._encode_on_tt

        def encode_with_pcc(input_ids, attention_mask):
            out = orig_encode(input_ids, attention_mask)
            # The pipeline consumes hidden_states[-2]; the twin runs the plain
            # HF path (no forced mask tracing) so it is a clean reference.
            golden = _twin(ModelVariant.TEXT_ENCODER).encoder(
                input_ids=_to_golden(input_ids),
                attention_mask=_to_golden(attention_mask),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            _gate("text_encoder", out, golden.hidden_states[-2])
            return out

        pipeline._encode_on_tt = encode_with_pcc

    # ── Lumina2Transformer2DModel (DiT) ────────────────────────────────
    if "transformer" in components:
        transformer = pipeline.transformer
        orig_transformer_forward = transformer.forward

        def transformer_with_pcc(
            hidden_states, timestep, encoder_hidden_states, encoder_attention_mask
        ):
            out = orig_transformer_forward(
                hidden_states, timestep, encoder_hidden_states, encoder_attention_mask
            )
            # NOTE: the caption length here (a real prompt fills only some of the
            # 256 slots, so the joint sequence is caption_len+4096 and generally
            # tile-unaligned) is NOT what this gate is sensitive to -- the DiT
            # passes 0.99 on a single executable at every caption length measured.
            # What this gate actually covers is the pipeline's *compilation path*:
            # it is the only place the DiT runs outside run_graph_test, so it is
            # the only thing that would catch the composite pass not being applied
            # (see LuminaImageConfig.compile_on_tt, which exists because of
            # exactly that regression).
            #
            # With ``pad_negative_caption`` on, the uncond forward is padded to the
            # positive's caption length, so both CFG forwards land here on one
            # executable. Turning it off splits them into two and trips the open
            # decay bug -- see LuminaImageConfig.pad_negative_caption.
            golden = _twin(ModelVariant.TRANSFORMER)(
                _to_golden(hidden_states),
                _to_golden(timestep),
                _to_golden(encoder_hidden_states),
                _to_golden(encoder_attention_mask),
            )
            _gate("transformer", out, golden)
            return out

        transformer.forward = transformer_with_pcc

    # ── AutoencoderKL decoder ──────────────────────────────────────────
    if "vae" in components:
        vae = pipeline.vae
        orig_vae_forward = vae.forward

        def vae_with_pcc(z):
            out = orig_vae_forward(z)
            golden = _twin(ModelVariant.VAE)(_to_golden(z))
            _gate("vae", out, golden)
            return out

        vae.forward = vae_with_pcc

    return history


def run_lumina_image_pipeline(
    output_path: str = OUTPUT_PATH,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
    shard: bool = True,
    check_pcc: bool = True,
    pcc_components=PCC_COMPONENTS,
    pcc_threshold: float = PCC_THRESHOLD,
):
    """Run the Lumina-Image-2.0 pipeline (all components on TT) and save image.

    Runs the pipeline's own defaults throughout — including ``cfg_trunc_ratio``
    (1.0, CFG on every step) and ``pad_negative_caption``. Returns
    ``(output_path, pcc_history)``; ``pcc_history`` is empty when ``check_pcc``
    is False.
    """
    pipeline = LuminaImagePipeline(
        config=LuminaImageConfig(
            on_tt=True,
            shard=shard,
            num_inference_steps=num_inference_steps,
        )
    )
    pipeline.setup()

    pcc_history: dict = {}
    if check_pcc:
        pcc_history = _attach_pcc_checks(
            pipeline,
            components=pcc_components,
            threshold=pcc_threshold,
        )

    # ``generate`` post-processes via the diffusers image processor and returns a
    # list of PIL images (output_type="pil").
    images = pipeline.generate(prompt=PROMPT, seed=SEED)
    images[0].save(output_path)
    return output_path, pcc_history


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
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

    DiT sharded on TT; text encoder, VAE, sampling and scheduler on CPU.

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
