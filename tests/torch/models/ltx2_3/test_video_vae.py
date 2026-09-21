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

Weights are real only where the 22B checkpoint is on the host. Where it is not,
``loader.checkpoint_path()`` returns None, the component runs on random weights
with identity ``PerChannelStatistics``, and ``model.weights_loaded`` is False --
the PCC then measures CPU-vs-TT numerics on the same graph, which is a valid
device-numerics result but NOT a statement about the trained model. The flag is
recorded with every rung metric below for exactly that reason.

That distinction decides whether a rung runs at all. The smoke tests are kept
as a device-numerics signal either way, but a PARITY rung on random weights
measures nothing about parity, so the ladder SKIPS rather than reporting a
number that reads like one (2026-09-17: a full ladder sweep was run before the
statistics repair existed, against uninitialized ``std-of-means``, and every
PCC and every NaN it produced had to be thrown away).

Two shapes appear below and they are NOT interchangeable:

* the SMOKE shape (9 frames at 256x256, the loader defaults) — the minimum
  round-trip that exercises both halves, kept as the fast CI signal;
* the PARITY LADDER — rungs climbing toward the reference configuration of 121
  frames at 512x768 (``loader.REFERENCE_CONFIG["standard"]``), which is what
  the pipeline actually runs.

Ladder rungs are given in VIDEO space and drive both halves from one spec: the
decoder's latent shape is derived by ``loader.video_latent_shape``, so the two
halves of a rung remain exact round-trip inverses. Frames and resolution climb
as SEPARATE ladders because they load different limits — temporal sequence
length vs per-frame activation footprint — and a single combined ladder cannot
say which one was hit.
"""

import json
import os
import time

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, RunMode, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from utils import BringupStatus, Category

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.ltx2_3.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.ltx2_3.pytorch.loader import checkpoint_path

# Set LTX23_LADDER_METRICS=1 to also record the per-rung numbers the parity
# matrix is built from (an extra CPU forward, so it is off in CI). The harness
# runs the CPU golden internally, so timing it separately is the only way to
# split golden cost out of the wall clock.
_COLLECT_METRICS = os.environ.get("LTX23_LADDER_METRICS", "0") == "1"

_REQUIRED_PCC = 0.99


def _properties(variant: ModelVariant, bringup_status: BringupStatus):
    return pytest.mark.record_test_properties(
        category=Category.MODEL_TEST,
        model_info=ModelLoader.get_model_info(variant),
        run_mode=RunMode.INFERENCE,
        bringup_status=bringup_status,
        pcc=_REQUIRED_PCC,
    )


# ----- Smoke tests (loader defaults: 9 frames at 256x256) -----


@pytest.mark.nightly
@pytest.mark.single_device
@_properties(ModelVariant.VIDEO_VAE_DECODER, BringupStatus.PASSED)
def test_ltx2_3_video_vae_decoder():
    _run(ModelVariant.VIDEO_VAE_DECODER)


@pytest.mark.nightly
@pytest.mark.single_device
@_properties(ModelVariant.VIDEO_VAE_ENCODER, BringupStatus.PASSED)
def test_ltx2_3_video_vae_encoder():
    _run(ModelVariant.VIDEO_VAE_ENCODER)


# ----- Parity ladder -----

_FRAME_LADDER = (25, 49, 121)  # at 256x256; 9 frames is the smoke test above
_RESOLUTION_LADDER = ((512, 512), (512, 768))  # at 9 frames
_REFERENCE_RUNG = (121, 512, 768)  # both axes at reference simultaneously

_HALVES = (ModelVariant.VIDEO_VAE_DECODER, ModelVariant.VIDEO_VAE_ENCODER)
_HALF_IDS = {
    ModelVariant.VIDEO_VAE_DECODER: "decoder",
    ModelVariant.VIDEO_VAE_ENCODER: "encoder",
}

# Rungs with an established device ceiling, keyed by param id. Only capacity
# failures belong here: the allocation is driven by the rung's shapes, so it
# reproduces whatever the weight values are, which is NOT true of anything the
# 2026-09-17 sweep reported as a numerics failure. Non-strict (pytest.ini sets
# no xfail_strict), so a plain run reports XFAIL -- use --runxfail to see the
# error itself.
_CEILINGS = {
    "decoder-121f-512x768-reference": (
        "DRAM OOM at the reference rung (2026-09-17, single device): "
        "TT_FATAL Out of Memory: Not enough space to allocate 1560281088 B "
        "DRAM buffer across 12 banks, where each bank needs to store "
        "130023424 B, but bank size is 1070773184 B "
        "(allocated: 817064864 B, free: 253708320 B, "
        "largest free block: 128186656 B); surfaces as "
        "RuntimeError: Bad StatusOr access: INTERNAL: Error code: 13"
    ),
}


def _marks(rung_id: str):
    reason = _CEILINGS.get(rung_id)
    return [pytest.mark.xfail(reason=reason)] if reason else []


def _rungs():
    for variant in _HALVES:
        half = _HALF_IDS[variant]
        for num_frames in _FRAME_LADDER:
            rung_id = f"{half}-{num_frames}f-256x256"
            yield pytest.param(
                variant,
                num_frames,
                256,
                256,
                id=rung_id,
                marks=_marks(rung_id),
            )
        for height, width in _RESOLUTION_LADDER:
            rung_id = f"{half}-9f-{height}x{width}"
            yield pytest.param(
                variant,
                9,
                height,
                width,
                id=rung_id,
                marks=_marks(rung_id),
            )
        rung_id = (
            f"{half}-{_REFERENCE_RUNG[0]}f-"
            f"{_REFERENCE_RUNG[1]}x{_REFERENCE_RUNG[2]}-reference"
        )
        yield pytest.param(
            variant,
            *_REFERENCE_RUNG,
            id=rung_id,
            marks=_marks(rung_id),
        )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.parametrize("variant,num_frames,height,width", list(_rungs()))
def test_ltx2_3_video_vae_parity_ladder(
    variant: ModelVariant, num_frames: int, height: int, width: int
):
    """One rung of the climb from the smoke shape to 121 frames at 512x768.

    A rung whose ceiling is already established carries an xfail holding the
    verbatim device error (see ``_CEILINGS``), so the ceiling stays visible
    instead of being hidden by keeping only the shape that happens to pass.
    """
    _run(
        variant,
        num_frames=num_frames,
        height=height,
        width=width,
        parity=True,
    )


def _run(
    variant: ModelVariant,
    *,
    num_frames=None,
    height=None,
    width=None,
    parity=False,
):
    xr.set_device_type("TT")
    torch.manual_seed(42)
    compiler_config = CompilerConfig(optimization_level=1)

    loader = ModelLoader(variant)
    t0 = time.perf_counter()
    model = loader.load_model(dtype_override=torch.bfloat16).eval()
    weights_loaded = bool(getattr(model, "weights_loaded", False))
    if parity and not weights_loaded:
        # Reporting a PCC here would read as a parity result for the trained
        # VAE, which is the exact mistake the 2026-09-17 sweep made.
        pytest.skip(
            "parity rung needs the real 22B checkpoint; "
            f"{checkpoint_path()!r} resolved to no file "
            "(set LTX2_3_CHECKPOINT). The smoke tests still cover "
            "CPU-vs-TT numerics on random weights."
        )
    inputs = loader.load_inputs(
        dtype_override=torch.bfloat16,
        num_frames=num_frames,
        height=height,
        width=width,
    )
    build_s = time.perf_counter() - t0

    cpu_s = None
    if _COLLECT_METRICS:
        t0 = time.perf_counter()
        with torch.no_grad():
            cpu_out = model(*inputs)
        cpu_s = time.perf_counter() - t0
        assert torch.isfinite(cpu_out).all(), "CPU golden contains NaN/Inf"
        del cpu_out

    t0 = time.perf_counter()
    try:
        run_graph_test(
            model,
            inputs,
            framework=Framework.TORCH,
            compiler_config=compiler_config,
            comparison_config=ComparisonConfig(
                pcc=PccConfig(required_pcc=_REQUIRED_PCC)
            ),
        )
    finally:
        if _COLLECT_METRICS:
            # One grep-able line per rung; cpu_s is the golden cost, so
            # tt_s covers compile + device execution + the PCC compare.
            print(
                "LTX23_LADDER_METRIC "
                + json.dumps(
                    {
                        "variant": str(variant),
                        "num_frames": num_frames if num_frames is not None else 9,
                        "height": height if height is not None else 256,
                        "width": width if width is not None else 256,
                        "input_shape": list(inputs[0].shape),
                        # Without this flag a metric row cannot be told apart
                        # from a real-weight one after the fact.
                        "weights_loaded": weights_loaded,
                        "build_s": round(build_s, 2),
                        "cpu_golden_s": round(cpu_s, 2) if cpu_s else None,
                        "tt_s": round(time.perf_counter() - t0, 2),
                    }
                ),
                flush=True,
            )
