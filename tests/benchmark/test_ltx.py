# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for the LTX-2.3 (Lightricks) audio-video DiT denoiser.

LTX-2.3 is a joint audio-video diffusion foundation model. The HuggingFace repo
ships a single monolithic checkpoint; only the on-checkpoint components are
exposed by the tt-forge-models loader, and the ~21B DiT denoiser
(``LTX2VideoTransformer3DModel``) is the per-step compute and the sharding
target. The text encoder/tokenizer live externally and the VAEs/vocoder are not
exposed, so this benchmark times **only the DiT transformer** — one denoising
forward pass on synthetic latents at native generation geometry
(512x768, 121 frames @ 24 fps -> 6144 video tokens, 376 audio tokens).

The DiT is too large for a single 32 GB Blackhole chip (~44 GB bf16), so it runs
under SPMD tensor-parallel sharding (Megatron column->row) across the device
mesh. The mesh shape and the weight shard specs are read from the loader's
``get_mesh_config`` / ``load_shard_spec`` hooks — the same hooks the model-bringup
baseline established — so the perf path inherits the sharded baseline topology.

This mirrors ``test_wan.py::test_wan_dit`` (the Wan 2.2 DiT benchmark): a single
component, list-of-tensors inputs, SPMD sharding via an
``apply_sharding_fn(wrapper, mesh)`` callback, driven through the shared video-gen
harness in ``benchmarks/video_gen_benchmark.py``. As with the Wan DiT, the run is
perf-only (``required_pcc=None``): a full CPU golden forward of a ~21B model is
impractical, so correctness is validated by the model-bringup HW test, not here.
"""

import json

import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from infra.utilities.torch_multichip_utils import get_mesh
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig

# Bringup-safe defaults; model-perf-tuning ramps these for headline numbers.
OPTIMIZATION_LEVEL = 0
TRACE_ENABLED = False


def test_ltx_2_3_dit(output_file, request):
    """Benchmark one LTX-2.3 DiT denoising forward pass under TP sharding."""
    from third_party.tt_forge_models.ltx_2_3.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    loader = ModelLoader(variant=ModelVariant.TRANSFORMER)
    # load_model returns an (eval, bf16) LTX2TransformerWrapper on CPU; load_inputs
    # returns the positional tensor list matching the wrapper's forward signature.
    wrapper = loader.load_model()
    inputs = loader.load_inputs()

    def mesh_fn():
        """Build the loader's ("batch", "model") mesh for the current devices."""
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        return get_mesh(mesh_shape, mesh_names)

    def apply_sharding_fn(wrapper_on_device, mesh):
        """Mark the loader's Megatron column->row TP specs on the device weights."""
        specs = loader.load_shard_spec(wrapper_on_device)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    compiler_config = CompilerConfig(
        optimization_level=OPTIMIZATION_LEVEL,
        enable_trace=TRACE_ENABLED,
    )

    model_info_name = "LTX-2.3-DiT"
    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print(f"Running LTX-2.3 DiT benchmark (sharded tensor-parallel)")

    results = benchmark_video_gen_torch_xla(
        wrapper=wrapper,
        inputs=inputs,
        model_info_name=model_info_name,
        display_name=display_name,
        compiler_config=compiler_config,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        sharded=True,
        mesh_fn=mesh_fn,
        apply_sharding_fn=apply_sharding_fn,
        # Perf-only: a full CPU golden of the ~21B DiT is impractical; correctness
        # is covered by the model-bringup HW test (mirrors test_wan_dit).
        required_pcc=None,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
