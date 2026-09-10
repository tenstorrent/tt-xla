# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Performance benchmarks for the LongCat-Image text-to-image pipeline.

``meituan-longcat/LongCat-Image`` is a bilingual text-to-image MMDiT
(~14 B aggregate) that is brought up **per component**, mirroring the functional
component tests in ``tests/torch/models/longcat_image/``:

  - ``test_longcat_text_encoder``  — Qwen2.5-VL 7B, one prompt encode per image.
  - ``test_longcat_transformer``   — ~6 B Flux-style MMDiT (10 dual + 20 single
    blocks); this is the per-denoising-step cost and therefore the headline
    number for the model.
  - ``test_longcat_vae_decoder``   — AutoencoderKL, one latent decode per image.

Neither heavy component fits a single n150, so both run tensor-parallel on an
FSDP-style ("batch", "model") mesh; the shard specs and the mesh shape come from
the ``tt_forge_models`` loader (``load_shard_spec`` / ``get_mesh_config``), which
is the same wiring the functional tests use. The mesh is derived from the
runtime device count, so the same test body covers a 4-chip quietbox ((1, 4))
and an 8-chip llmbox ((2, 4)) — the "model" axis is capped at 4 by the text
encoder's GQA (4 KV heads). The VAE decoder fits one chip and runs unsharded.

Each component is measured as a single forward through
``benchmarks/video_gen_benchmark.py`` (a list of input tensors, optional SPMD
sharding, per-forward latency) and is PCC-gated against a CPU golden at the same
0.99 threshold the functional tests use, so a numerics regression fails the
benchmark instead of quietly reporting a fast wrong answer.

NOTE: shapes come from the loader's captured I/O spec, which pins the pipeline
at 256x256 (latent sequence 256, text sequence 512). The reported per-forward
latency is therefore for that resolution, not LongCat's native 1024x1024;
raising it requires the loader to rebuild its pinned ``img_ids`` position
buffers for the larger patch grid.
"""

import json

import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from infra.utilities.torch_multichip_utils import get_mesh
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.longcat_image.pytorch import ModelLoader, ModelVariant

SEED = 42
DATA_FORMAT = torch.bfloat16

# Compile options per component. These must stay in sync with the
# ``COMPILER_CONFIG`` of the matching functional test in
# ``tests/torch/models/longcat_image/`` so the PCC gate there and the perf
# number here always compile the component identically. All three are currently
# plain defaults; if a component ever needs an override, change it in both
# places.
TEXT_ENCODER_COMPILER_CONFIG = CompilerConfig()
TRANSFORMER_COMPILER_CONFIG = CompilerConfig()
VAE_COMPILER_CONFIG = CompilerConfig()

# Same gate as the functional component tests (infra ``PccConfig`` default), so
# the benchmark cannot report a number for a run the functional suite would fail.
REQUIRED_PCC = 0.99

MODEL_TYPE = "Image Generation, Text-to-Image"


def _mesh_fn(loader):
    """Build the loader's ("batch", "model") mesh for the visible device count."""

    def build_mesh():
        mesh_shape, mesh_names = loader.get_mesh_config(
            xr.global_runtime_device_count()
        )
        return get_mesh(mesh_shape, mesh_names)

    return build_mesh


def _sharding_fn(loader):
    """Mark the loader's tensor-parallel weight specs on the on-device wrapper.

    ``load_shard_spec`` takes the module returned by ``load_model`` and returns
    ``{param -> partition_spec}``; the harness calls this after the wrapper is
    moved to the device, so the parameters it walks are already XLA tensors.
    """

    def apply_sharding(wrapper, mesh):
        specs = loader.load_shard_spec(wrapper)
        for tensor, spec in (specs or {}).items():
            xs.mark_sharding(tensor, mesh, spec)

    return apply_sharding


def _run_longcat_benchmark(
    *,
    variant,
    model_info_name,
    compiler_config,
    sharded,
    output_file,
    request,
):
    """Load one LongCat-Image component, benchmark it, and persist the JSON result."""
    torch.manual_seed(SEED)

    loader = ModelLoader(variant)
    wrapper = loader.load_model(dtype_override=DATA_FORMAT).eval()
    inputs = loader.load_inputs(dtype_override=DATA_FORMAT)

    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print(f"Running LongCat-Image benchmark: {model_info_name} (sharded={sharded})")

    results = benchmark_video_gen_torch_xla(
        wrapper=wrapper,
        inputs=inputs,
        model_info_name=model_info_name,
        display_name=display_name,
        compiler_config=compiler_config,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        sharded=sharded,
        mesh_fn=_mesh_fn(loader) if sharded else None,
        apply_sharding_fn=_sharding_fn(loader) if sharded else None,
        required_pcc=REQUIRED_PCC,
        model_type=MODEL_TYPE,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)


def test_longcat_text_encoder(output_file, request):
    """Qwen2.5-VL 7B prompt encode — tensor-parallel (OOMs on a single chip)."""
    _run_longcat_benchmark(
        variant=ModelVariant.TEXT_ENCODER,
        model_info_name="LongCat-Image-Text-Encoder",
        compiler_config=TEXT_ENCODER_COMPILER_CONFIG,
        sharded=True,
        output_file=output_file,
        request=request,
    )


def test_longcat_transformer(output_file, request):
    """~6 B MMDiT — one denoising step, tensor-parallel (OOMs on a single chip)."""
    _run_longcat_benchmark(
        variant=ModelVariant.TRANSFORMER,
        model_info_name="LongCat-Image-Transformer",
        compiler_config=TRANSFORMER_COMPILER_CONFIG,
        sharded=True,
        output_file=output_file,
        request=request,
    )


def test_longcat_vae_decoder(output_file, request):
    """AutoencoderKL latent -> image decode; fits a single chip, unsharded."""
    _run_longcat_benchmark(
        variant=ModelVariant.VAE,
        model_info_name="LongCat-Image-VAE-Decoder",
        compiler_config=VAE_COMPILER_CONFIG,
        sharded=False,
        output_file=output_file,
        request=request,
    )
