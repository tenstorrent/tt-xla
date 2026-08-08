# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""HunyuanVideo — nightly e2e text-to-video pipeline test with per-step PCC checks.

HunyuanVideo is a diffusion text-to-video model whose DiT transformer runs
tensor-parallel across a multi-chip mesh, while the LLaMA-3 and CLIP text
encoders, the FlowMatchEuler scheduler and the VAE decode stay on CPU. This
drives the shared HunyuanVideoPipeline from ``tt_forge_models`` (the same
pipeline the video-gen benchmark uses) end-to-end, gates its numerics per
denoising step and asserts the saved frame dimensions.

The test fails fast the moment any DiT step drops below ``PCC_THRESHOLD``. The
pipeline itself keeps using the TT outputs.

The DiT is the only component that runs on TT in this pipeline — the LLaMA and
CLIP text encoders and the VAE all run on CPU here.
"""

from pathlib import Path

import pytest
import torch
import torch_xla.runtime as xr
from diffusers.utils import export_to_video
from infra import RunMode
from infra.evaluators import PccConfig, TorchComparisonEvaluator
from infra.evaluators.evaluation_config import ComparisonConfig
from loguru import logger
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.hunyuan_video.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.hunyuan_video.pytorch.src.pipeline import (
    HEIGHT,
    WIDTH,
    HunyuanVideoConfig,
    HunyuanVideoPipeline,
)

PROMPT = (
    "A fluffy orange tabby cat walks slowly across a lush green meadow on a "
    "bright sunny morning, soft golden sunlight, gentle breeze moving the "
    "grass and the cat's fur, shallow depth of field, cinematic, photorealistic, "
    "highly detailed, smooth natural motion"
)

VARIANT_NAME = ModelVariant.TRANSFORMER
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

SEED = 42
NUM_INFERENCE_STEPS = 30
# Full-length clip matching the HunyuanVideo model card's diffusers example
# (num_frames=61 @ fps 15 => ~4s). HunyuanVideo's VAE compresses time 4x, so this
# is 16 latent frames -- much heavier on the DiT than the single-frame smoke run.
# https://huggingface.co/hunyuanvideo-community/HunyuanVideo
NUM_FRAMES = 61
# bf16 DiT on TT vs a clean fp32 CPU reference. Mirrors tt-xla#5480's 0.99 gate;
# tune against the first nightly if the bf16/fp32 gap sits below this.
PCC_THRESHOLD = 0.99


_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()


def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))


def _attach_dit_pcc_check(pipeline: HunyuanVideoPipeline) -> None:
    """Wrap the pipeline's DiT forward with an inline fp32-CPU-twin PCC check.

    Only the DiT transformer runs on TT, so it is the sole component gated here.
    After each TT forward the same inputs are replayed on a lazily loaded fp32
    CPU twin and PCC is asserted inline, so the test fails fast on the first
    diverging denoising step (checked per DiT forward; HunyuanVideo is
    guidance-distilled so the default path is one forward per step).

    Must be called after ``pipeline.setup()`` — setup shards the DiT, moves it to
    the XLA device and installs the ``torch.compile(backend="tt")`` forward, and
    ``pipeline.transformer`` is the object ``generate()`` invokes.
    """
    transformer = pipeline.transformer
    # The compiled (backend="tt") forward that setup() installed as an instance
    # attribute; captured before the wrapper below shadows it, so the TT path
    # still runs through Dynamo.
    orig_forward = transformer.forward

    twin = {"model": None}
    step = {"n": 0}

    def _cpu_twin():
        # Loaded on first use: a fresh fp32 CPU copy of the DiT. It stays plain
        # eager fp32 (no bf16 cast, no torch.compile) so it is a clean golden
        # reference for the bf16 compiled TT DiT. ModelLoader returns the
        # HunyuanVideoTransformerWrapper (positional tensor-only forward).
        if twin["model"] is None:
            logger.info("[PCC] loading CPU fp32 DiT twin")
            twin["model"] = ModelLoader(ModelVariant.TRANSFORMER).load_model(
                dtype_override=torch.float32
            )
        return twin["model"]

    def _cpu(x):
        # Move to CPU and upcast floats to fp32 so the twin sees the same values
        # the TT DiT consumed; leave int/bool tensors untouched.
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
        # the DiT with these keywords (see HunyuanVideoPipeline.generate._dit);
        # the wrapper twin takes them positionally.
        golden_sample = _cpu_twin()(
            _cpu(kwargs["hidden_states"]),
            _cpu(kwargs["timestep"]),
            _cpu(kwargs["encoder_hidden_states"]),
            _cpu(kwargs["encoder_attention_mask"]),
            _cpu(kwargs["pooled_projections"]),
            _cpu(kwargs["guidance"]),
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
    """Run the HunyuanVideo pipeline (DiT tensor-parallel) with per-step DiT PCC."""
    xr.set_device_type("TT")

    pipeline = HunyuanVideoPipeline(
        config=HunyuanVideoConfig(
            num_inference_steps=NUM_INFERENCE_STEPS, num_frames=NUM_FRAMES
        )
    )
    pipeline.setup()

    # Gate the DiT (the only TT component) against an fp32 CPU twin per step.
    _attach_dit_pcc_check(pipeline)

    # ``generate`` post-processes via the diffusers video processor and returns a
    # list (one per batch) of lists of PIL frames (output_type="pil").
    video = pipeline.generate(prompt=PROMPT, seed=SEED)
    frames = video[0]

    # Every generated frame must match the requested spatial dimensions.
    assert len(frames) == NUM_FRAMES, f"Expected {NUM_FRAMES} frames, got {len(frames)}"
    for i, frame in enumerate(frames):
        width, height = frame.size
        assert width == WIDTH, f"Frame {i}: expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Frame {i}: expected height {HEIGHT}, got {height}"

    # Encode the frames to an .mp4 video (fps 15, matching the reference example).
    output_path = "hunyuan_video_pipeline_output_aug3.mp4"
    export_to_video(frames, output_path, fps=15)
    assert Path(output_path).exists(), f"Output video {output_path} was not created"
