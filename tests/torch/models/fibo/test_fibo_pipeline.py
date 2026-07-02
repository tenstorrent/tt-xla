# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FIBO (briaai/FIBO) — nightly e2e text-to-image pipeline test + perf harness.

FIBO is an 8B-parameter DiT text-to-image model whose transformer runs out of
DRAM on a single Wormhole chip, so the heavy net runs **tensor-parallel** across
a multi-chip mesh (Megatron-1D), while the SmolLM3 text encoder, scheduler and
Wan 2.2 VAE stay on CPU. This drives the shared ``FiboPipeline`` from
``tt_forge_models`` (the same pipeline the image-gen benchmark uses) end-to-end
and asserts the saved image dimensions.

In addition to the correctness check, this script measures **how quickly a
single image can be generated after a few warmup runs**. The first
``generate()`` pays the one-time ``torch.compile`` + device-program build cost;
warmup runs absorb that so the measured runs reflect steady-state latency. Each
run's timing is split into the on-device tensor-parallel denoise (summed from
the pipeline's per-step ``_perf["steps"]``) and the CPU-side remainder (text
encode + VAE decode), then summarized as per-image latency and throughput
(images/s).

Tunables (env):
  FIBO_WARMUP_RUNS   warmup generations, timings discarded   (default 1)
  FIBO_MEASURE_RUNS  timed generations used for the summary   (default 3)
"""

import os
import statistics
import time
from pathlib import Path

import pytest
import torch
from infra import RunMode
from PIL import Image
from utils import BringupStatus, Category

from third_party.tt_forge_models.config import Parallelism
from third_party.tt_forge_models.fibo.pytorch import ModelLoader, ModelVariant

# The FIBO e2e pipeline lives in tt-forge-models; skip cleanly until the
# submodule uplift brings in fibo/pytorch/pipeline.py (this PR carries no bump).
_pipeline = pytest.importorskip(
    "third_party.tt_forge_models.fibo.pytorch.pipeline",
    reason="requires tt-forge-models fibo/pytorch/pipeline.py (submodule uplift)",
)
FiboConfig = _pipeline.FiboConfig
FiboPipeline = _pipeline.FiboPipeline

VARIANT_NAME = ModelVariant.BASE
MODEL_INFO = ModelLoader._get_model_info(VARIANT_NAME)

PROMPT = (
    '{"subject":"a hyper-detailed, ultra-fluffy owl perched in moonlit trees",'
    '"style_medium":"photograph","camera":"85mm prime, shallow depth of field",'
    '"lighting":"cool moonlight with subtle silver highlights"}'
)
NUM_INFERENCE_STEPS = 50
SEED = 42
HEIGHT = 1024
WIDTH = 1024

# A single image after a couple of warmups is enough to read steady-state
# latency; both are env-overridable so the harness runtime can be tuned.
WARMUP_RUNS = int(os.environ.get("FIBO_WARMUP_RUNS", "1"))
MEASURE_RUNS = int(os.environ.get("FIBO_MEASURE_RUNS", "3"))


def _timed_generate(pipeline, tag):
    """Run one ``generate()``, print a per-run breakdown, return (image, wall, device)."""
    t0 = time.perf_counter()
    image = pipeline.generate(
        prompt=PROMPT,
        num_inference_steps=NUM_INFERENCE_STEPS,
        seed=SEED,
    )
    wall = time.perf_counter() - t0

    perf = pipeline._perf or {}
    steps = perf.get("steps") or []
    device_denoise = float(sum(steps))
    n_steps = len(steps)
    per_step = device_denoise / n_steps if n_steps else float("nan")
    cpu_overhead = wall - device_denoise  # text-encode + VAE decode (both on CPU)

    print(
        f"[{tag}] wall={wall:.2f}s | device denoise={device_denoise:.2f}s "
        f"({n_steps} steps, {per_step:.2f}s/step) | cpu overhead={cpu_overhead:.2f}s",
        flush=True,
    )
    return image, wall, device_denoise


@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.llmbox
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_info=MODEL_INFO,
    parallelism=Parallelism.TENSOR_PARALLEL,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_fibo_pipeline():
    """Run FIBO (DiT tensor-parallel), measure single-image gen perf, assert image."""
    pipeline = FiboPipeline(config=FiboConfig())
    pipeline.setup()

    # Warmup: the first generate() triggers torch.compile + the device program
    # build; a couple of runs stabilize caches. Timings are discarded.
    for i in range(WARMUP_RUNS):
        image, _, _ = _timed_generate(pipeline, f"warmup {i + 1}/{WARMUP_RUNS}")

    # Measured runs: steady-state single-image generation.
    walls, denoises = [], []
    for i in range(MEASURE_RUNS):
        image, wall, device_denoise = _timed_generate(
            pipeline, f"measure {i + 1}/{MEASURE_RUNS}"
        )
        walls.append(wall)
        denoises.append(device_denoise)

    mean_wall = statistics.mean(walls)
    min_wall = min(walls)
    stdev_wall = statistics.pstdev(walls) if len(walls) > 1 else 0.0
    mean_denoise = statistics.mean(denoises)

    print(
        "\n================ FIBO single-image generation perf ================\n"
        f"config          : {WIDTH}x{HEIGHT}, {NUM_INFERENCE_STEPS} steps, "
        f"tensor-parallel DiT on mesh (SmolLM3 + Wan VAE on CPU)\n"
        f"runs            : {WARMUP_RUNS} warmup (discarded) + {MEASURE_RUNS} measured\n"
        f"per-image latency: mean={mean_wall:.2f}s  min={min_wall:.2f}s  "
        f"stdev={stdev_wall:.2f}s\n"
        f"  device denoise : mean={mean_denoise:.2f}s "
        f"({mean_denoise / NUM_INFERENCE_STEPS:.2f}s/step)\n"
        f"  cpu overhead   : mean={mean_wall - mean_denoise:.2f}s "
        f"(text encode + VAE decode)\n"
        f"throughput      : mean={1.0 / mean_wall:.4f} images/s  "
        f"best={1.0 / min_wall:.4f} images/s\n"
        "===================================================================\n",
        flush=True,
    )

    output_path = "test_fibo_pipeline_output.png"
    array = (image[0].float().clamp(0, 1) * 255).round().to(torch.uint8)
    array = array.permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(output_path)

    assert Path(output_path).exists(), f"Output image {output_path} was not created"
    with Image.open(output_path) as img:
        width, height = img.size
        assert width == WIDTH, f"Expected width {WIDTH}, got {width}"
        assert height == HEIGHT, f"Expected height {HEIGHT}, got {height}"
