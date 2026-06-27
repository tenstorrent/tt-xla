# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for the LTX-2 (Lightricks) audiovisual video DiT denoiser.

LTX-2 is an audiovisual text/image-to-video diffusion pipeline built around a
~19B audiovisual DiT denoiser (``LTX2VideoTransformer3DModel``). The pipeline's
scheduler / denoising loop / latent glue stay in host Python; the DiT is the
per-step compute and the sharding target, so this benchmark times **only the DiT
transformer** — one denoising forward pass on synthetic packed latents at native
generation geometry (512x768, 121 frames @ 24 fps -> 6144 video tokens, 126
audio tokens), with the audio + video cross-attention modalities both active.

The DiT is too large for a single 32 GB Blackhole chip (~38 GB bf16), so it runs
under SPMD tensor-parallel sharding (Megatron column->row) across the device
mesh. The mesh shape and the weight shard specs are read from the loader's
``get_mesh_config`` / ``load_shard_spec`` hooks — the same hooks the model-bringup
baseline established — so the perf path inherits the sharded baseline topology
(for 4 chips the mesh is ``(1, 4)``).

This mirrors ``test_wan.py::test_wan_dit`` and the LTX-2.3 DiT benchmark: a single
component, list-of-tensors inputs, SPMD sharding via an
``apply_sharding_fn(wrapper, mesh)`` callback, driven through the shared video-gen
harness in ``benchmarks/video_gen_benchmark.py``. As with the Wan DiT, the run is
perf-only (``required_pcc=None``): a full CPU golden forward of a ~19B model is
impractical, so correctness is validated by the model-bringup HW test, not here.

Unlike the LTX-2.3 loader, the ``ltx_2`` loader returns the raw
``LTX2VideoTransformer3DModel`` (whose forward takes keyword args mixing tensors
with static latent-grid geometry) plus a kwargs dict, rather than an
``(nn.Module, positional-tensor-list)`` pair. A thin ``_LTX2DiTAdapter`` bridges
that to the harness's positional ``wrapper(*inputs)`` contract: the loader dict's
tensors become the positional input list (in a fixed order) and the remaining
static geometry is captured as constants on the adapter. The loader's in-graph
squeeze->reshape workaround (``_patch_squeeze_non_aliasing``, applied inside
``load_transformer``) sidesteps the ``prims::view_of`` functionalization gap.
"""

import json

import torch
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from infra.utilities.torch_multichip_utils import get_mesh
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig

# Bringup-safe defaults; model-perf-tuning ramps these for headline numbers.
OPTIMIZATION_LEVEL = 0
TRACE_ENABLED = False


class _LTX2DiTAdapter(torch.nn.Module):
    """Adapt the LTX-2 DiT's kwargs forward to the harness positional contract.

    The video-gen harness calls ``wrapper(*inputs)`` with a list of CPU tensors
    (each moved to device) and expects a single output tensor. The
    ``LTX2VideoTransformer3DModel`` forward instead takes keyword args mixing
    tensors with static latent-grid geometry (num_frames / height / width / fps /
    audio_num_frames / return_dict). We accept the tensors positionally in a fixed
    order (``tensor_keys``) and inject the captured static kwargs on every call.
    """

    def __init__(self, transformer, tensor_keys, static_kwargs):
        super().__init__()
        self.transformer = transformer
        self._tensor_keys = list(tensor_keys)
        self._static_kwargs = dict(static_kwargs)

    def forward(self, *tensors):
        kwargs = dict(zip(self._tensor_keys, tensors))
        kwargs.update(self._static_kwargs)
        out = self.transformer(**kwargs)
        # return_dict=False -> tuple; take the primary (video) sample tensor.
        return out[0] if isinstance(out, (tuple, list)) else out


def test_ltx_2_dit(output_file, request):
    """Benchmark one LTX-2 DiT denoising forward pass under TP sharding."""
    from third_party.tt_forge_models.ltx_2.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    loader = ModelLoader(variant=ModelVariant.LTX_2_19B, subfolder="transformer")
    # Raw LTX2VideoTransformer3DModel (eval, bf16) on CPU + a kwargs dict mixing
    # tensors with static latent-grid geometry.
    transformer = loader.load_model()
    raw_inputs = loader.load_inputs()

    # Partition the loader dict: tensors become the positional input list (fixed
    # order), the rest are captured as constants on the adapter.
    tensor_keys = [k for k, v in raw_inputs.items() if isinstance(v, torch.Tensor)]
    inputs = [raw_inputs[k] for k in tensor_keys]
    static_kwargs = {
        k: v for k, v in raw_inputs.items() if not isinstance(v, torch.Tensor)
    }
    wrapper = _LTX2DiTAdapter(transformer, tensor_keys, static_kwargs)

    def mesh_fn():
        """Build the loader's ("batch", "model") mesh for the current devices."""
        num_devices = xr.global_runtime_device_count()
        mesh_shape, mesh_names = loader.get_mesh_config(num_devices)
        return get_mesh(mesh_shape, mesh_names)

    def apply_sharding_fn(wrapper_on_device, mesh):
        """Mark the loader's Megatron column->row TP specs on the device weights.

        ``load_shard_spec`` walks the transformer's blocks, so it gets the bare
        transformer module, not the adapter wrapping it.
        """
        specs = loader.load_shard_spec(wrapper_on_device.transformer)
        for tensor, spec in specs.items():
            xs.mark_sharding(tensor, mesh, spec)

    compiler_config = CompilerConfig(
        optimization_level=OPTIMIZATION_LEVEL,
        enable_trace=TRACE_ENABLED,
    )

    model_info_name = "LTX-2-DiT"
    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    print("Running LTX-2 DiT benchmark (sharded tensor-parallel)")

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
        # Perf-only: a full CPU golden of the ~19B DiT is impractical; correctness
        # is covered by the model-bringup HW test (mirrors test_wan_dit).
        required_pcc=None,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
