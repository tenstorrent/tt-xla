# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmark for the FIBO (briaai/FIBO) VAE decoder component.

FIBO is BRIA AI's 8B DiT-based flow-matching text-to-image model. It reuses the
Wan 2.2 3D-causal VAE (``diffusers.AutoencoderKLWan``, z_dim=48, spatial stride
16) — the *same* architecture as the Wan 2.2 TI2V-5B VAE. Only the VAE
**decoder** is benchmarked here: it is the single-chip conv3d component FIBO
runs after the DiT to turn a denoised latent back into pixels, and it is brought
up independently of the 8B DiT (which needs a multi-chip TP harness — see the
``fibo`` loader's ``get_mesh_config`` / ``build_shard_spec``).

At FIBO's native 1024x1024 image resolution the decoder input latent is
``[1, 48, 1, 64, 64]`` (B, z_dim, latent_frames=1, H//16, W//16) and it decodes
to a pixel tensor ``[1, 3, 1, 1024, 1024]``. Unlike the Wan *video* VAE (21-31
latent frames), this is a single image frame, so it is a much lighter conv3d
workload — which is what lets it run on one chip.

Because the decoder is the Wan 2.2 VAE, it needs the two Wan-specific fixes from
``tests/torch/models/wan5b/monkey_patch.py`` (identical VAE class → identical
trace obstacles):
  - ``_patch_wan_resample_rep_sentinel`` — makes ``WanResample``'s ``"Rep"``
    string sentinel traceable (object identity instead of ``Tensor == str``),
    which otherwise graph-breaks dynamo;
  - ``safe_xla_slicing`` — normalizes out-of-range slice bounds that CPU clamps
    silently but torch-xla rejects (e.g. ``x[:, :, -2:, :, :]`` on a size-1
    temporal dim). It wraps the whole compile + execution region.

Compiler config: ``optimization_level=2``. Level 2 enables the memory-layout
(sharding) optimizations on top of level 1's conv fusions — the point at which
conv-heavy models like this VAE improve drastically over level 0/1. The rest of
the config mirrors the Wan VAE decoder's proven options
(``experimental_enable_dram_space_saving_optimization`` + ``enable_trace``).

The measurement is **perf-only** (``required_pcc=None``): a bf16 conv3d CPU
golden for this VAE is impractically slow on the CI host (PyTorch has no fast
CPU bf16 conv kernel), so — exactly like the Wan VAE decoder benchmark in
``test_wan.py`` — no CPU golden is built. Correctness of this VAE is validated
separately during model bringup against an fp32 golden.
"""

import json

from benchmarks.video_gen_benchmark import benchmark_video_gen_torch_xla
from utils import aggregate_ttnn_perf_metrics, resolve_display_name

from tests.infra.testers.compiler_config import CompilerConfig
from tests.torch.models.wan5b.monkey_patch import (
    _patch_wan_resample_rep_sentinel,
    safe_xla_slicing,
)

# Level 2 turns on the memory-layout / sharding optimizations that give
# conv-based models their large speedup; the DRAM space-saving pass + trace
# hoisting mirror the (proven) Wan 2.2 VAE decoder config, which shares this
# exact AutoencoderKLWan architecture.
COMPILER_CONFIG = CompilerConfig(
    optimization_level=2,
    experimental_enable_dram_space_saving_optimization=True,
    enable_trace=True,
    math_fidelity="hifi2",
)


def test_fibo_vae_decoder(output_file, request):
    """FIBO Wan-2.2 VAE decoder on a single chip (perf-only, opt_level=2).

    Decodes a native-1024x1024 latent ``[1, 48, 1, 64, 64]`` to pixels
    ``[1, 3, 1, 1024, 1024]``.
    """
    from third_party.tt_forge_models.fibo.pytorch.loader import (
        ModelLoader,
        ModelVariant,
    )

    # Apply the WanResample rep-sentinel patch here (not at import time) so it
    # does not leak into other benchmark tests collected in the same session,
    # and before building the decoder so its traced forward uses the patched
    # object-identity sentinel.
    _patch_wan_resample_rep_sentinel()

    loader = ModelLoader(ModelVariant.VAE_DECODER)
    # VAEDecoderWrapper(AutoencoderKLWan) — already eval() + bf16 from the loader.
    wrapper = loader.load_model()
    # Single-element list: the [1, 48, 1, 64, 64] bf16 latent at native res.
    inputs = loader.load_inputs()

    model_info_name = "FIBO-VAE-Decoder"
    display_name = resolve_display_name(request=request, fallback=model_info_name)
    ttnn_perf_metrics_output_file = f"tt_xla_{display_name}_perf_metrics"

    results = benchmark_video_gen_torch_xla(
        wrapper=wrapper,
        inputs=inputs,
        model_info_name=model_info_name,
        display_name=display_name,
        compiler_config=COMPILER_CONFIG,
        ttnn_perf_metrics_output_file=ttnn_perf_metrics_output_file,
        # Single p150 — run replicated on one chip, no SPMD sharding.
        sharded=False,
        # safe_xla_slicing wraps compile + every forward: AutoencoderKLWan does
        # slice indexing that torch-xla rejects unless the indices are first
        # normalized into range.
        compile_context=safe_xla_slicing,
        # Perf-only: skip the (prohibitively slow) bf16 conv3d CPU golden, same
        # as the Wan VAE decoder benchmark. Correctness is validated at bringup.
        required_pcc=None,
    )

    if output_file:
        results["project"] = "tt-forge/tt-xla"
        results["model_rawname"] = model_info_name
        aggregate_ttnn_perf_metrics(ttnn_perf_metrics_output_file, results)
        with open(output_file, "w") as file:
            json.dump(results, file, indent=2)
