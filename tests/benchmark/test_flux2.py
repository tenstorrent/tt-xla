# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for the FLUX.2-dev text-to-image pipeline.

FLUX.2-dev (`black-forest-labs/FLUX.2-dev`) is a rectified-flow text-to-image
model. Unlike a single causal LM, it is composed of three independently loadable
components, each benchmarked separately here:

  - test_flux2_dev_transformer   : Flux2Transformer2DModel (MM-DiT, ~32B) [TP]
  - test_flux2_dev_text_encoder  : Mistral3 prompt embedder  (~24B)       [TP]
  - test_flux2_dev_vae           : AutoencoderKLFlux2 decoder (~0.1B)      [single chip]

The two large components are sharded tensor-parallel across the device mesh
(qb2-blackhole = 4 chips); the VAE fits on a single chip. A fourth test,
test_flux2_dev_generate_image, runs the full pipeline end-to-end and saves a
sample PNG as a CI artifact to confirm the components together produce a
coherent image.
"""

import json
import os

import torch
from benchmarks.flux2_benchmark import benchmark_flux2_component_torch_xla
from utils import resolve_display_name

# Bringup-safe defaults (see model-perf-tuning for ramping these).
DEFAULT_OPTIMIZATION_LEVEL = 0
DEFAULT_TRACE_ENABLED = False
DEFAULT_DATA_FORMAT = torch.bfloat16

MODEL_INFO_NAME = "FLUX.2-dev"


def _run_component(
    variant_name,
    output_file,
    request,
    optimization_level,
    *,
    loop_count,
    warmup_steps,
    required_pcc,
    default_optimization_level=DEFAULT_OPTIMIZATION_LEVEL,
    trace_enabled=DEFAULT_TRACE_ENABLED,
    custom_compile_options=None,
):
    """Shared driver: load the flux2 loader for `variant_name` and benchmark it.

    `default_optimization_level` is the per-component tuned default (baked in
    after the tuning sweep); the `--optimization-level` CLI flag overrides it.
    """
    from third_party.tt_forge_models.flux2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    variant = ModelVariant(variant_name)
    loader = ModelLoader(variant=variant)

    opt_level = (
        optimization_level
        if optimization_level is not None
        else default_optimization_level
    )

    display_name = resolve_display_name(request=request, fallback=MODEL_INFO_NAME)
    model_info_name = f"{MODEL_INFO_NAME} ({variant_name})"

    results = benchmark_flux2_component_torch_xla(
        loader=loader,
        variant=variant,
        model_info_name=model_info_name,
        display_name=display_name,
        optimization_level=opt_level,
        trace_enabled=trace_enabled,
        loop_count=loop_count,
        warmup_steps=warmup_steps,
        required_pcc=required_pcc,
        data_format=DEFAULT_DATA_FORMAT,
        custom_compile_options=custom_compile_options,
    )

    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)


def test_flux2_dev_transformer(output_file, request, optimization_level):
    """MM-DiT (Flux2Transformer2DModel, ~32B) — tensor parallel across the mesh.

    One forward = one denoise step. PCC is measured against a CPU golden run on
    synthetic N(0,1) inputs; the threshold reflects bf16 accumulation across a
    32B model sharded over the mesh (see report notes).
    """
    _run_component(
        "Dev",
        output_file,
        request,
        optimization_level,
        loop_count=8,
        warmup_steps=2,
        required_pcc=0.80,
        # Tuned: optimization_level 2 gives ~11% speedup with no PCC regression
        # (0.837 vs 0.831 at level 0). See report.
        default_optimization_level=2,
    )


def test_flux2_dev_text_encoder(output_file, request, optimization_level):
    """Mistral3 prompt embedder (~24B) — tensor parallel across the mesh.

    Builds the 15360-dim conditioning sequence the DiT consumes (hidden states
    from layers 10/20/30 stacked).
    """
    _run_component(
        "Dev_TextEncoder",
        output_file,
        request,
        optimization_level,
        loop_count=8,
        warmup_steps=2,
        required_pcc=0.97,
        # Stays at optimization_level 0. Tuning sweep (2026-06-02): BOTH level 1
        # and level 2 collapse PCC to ~0.07 (level 1 would be +18% otherwise),
        # so the collapse threshold is opt>=1, not just opt 2. The collapse is
        # structural, not an accumulation-precision issue: fp32_dest_acc_en=true
        # + math_fidelity=hifi4 at opt 1 leaves PCC at the identical 0.0678.
        # See report.
        default_optimization_level=0,
    )


def test_flux2_dev_vae(output_file, request, optimization_level):
    """AutoencoderKLFlux2 decoder (~0.1B) — single chip (latent -> image)."""
    _run_component(
        "Dev_VAE",
        output_file,
        request,
        optimization_level,
        loop_count=8,
        warmup_steps=2,
        required_pcc=0.95,
        # Tuned: optimization_level 2 gives ~26% speedup, PCC stays high
        # (0.993 vs 0.9998 at level 0). trace_enabled was also swept and is
        # neutral here (+0.1%): a single VAE forward has nothing to amortize, so
        # trace stays off. See report.
        default_optimization_level=2,
    )


def _artifact_dir():
    """Resolve the directory CI uploads as artifacts (next to the perf report)."""
    workspace = os.environ.get("GITHUB_WORKSPACE")
    report_file = os.environ.get("REPORT_FILE")
    if workspace and report_file:
        return os.path.join(workspace, os.path.dirname(report_file))
    return os.getcwd()


def test_flux2_dev_generate_image(request):
    """Run the full FLUX.2-dev pipeline and save a sample image as a CI artifact.

    This exercises all three components together (text encoder -> DiT denoise ->
    VAE decode) and writes the resulting PNG so the bringup produces a visual,
    human-checkable confirmation of correctness. The pipeline runs on CPU
    (diffusers' Flux2Pipeline orchestrates the scheduler loop); the per-component
    device benchmarks above prove each component runs on multi-chip hardware.
    """
    from third_party.tt_forge_models.flux2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    loader = ModelLoader(variant=ModelVariant.DEV)
    image = loader.generate_image(
        prompt="A photorealistic astronaut riding a horse on the moon, "
        "earth in the sky, cinematic lighting",
        num_inference_steps=8,
        height=256,
        width=256,
        seed=0,
    )

    out_dir = _artifact_dir()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flux2-dev-sample.png")
    image.save(out_path)
    print(f"Saved FLUX.2-dev sample image artifact to: {out_path}")
    assert os.path.exists(out_path) and os.path.getsize(out_path) > 0
