# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 A14B — 3D Causal VAE Decoder component test (Wan 2.1 VAE).

Decodes a denoised latent back to pixel space.

IN:  z (1, 16, latent_frames, latent_h, latent_w)
OUT: sample (1, 3, num_frames, video_h, video_w)
"""

import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd

from tests.infra.testers.compiler_config import CompilerConfig

from .monkey_patch import (
    _patch_wan_resample_avoid_4d_fold,
    _patch_wan_resample_rep_sentinel,
    _disable_tt_torch_function_override,
    safe_xla_slicing,
)
from .shared import (
    LATENT_CHANNELS,
    RESOLUTIONS,
    VAEDecoderWrapper,
    compute_pcc,
    load_vae,
    shard_vae_decoder_specs,
    wan22_mesh,
)

# ---------------------------------------------------------------------------
# Monkey patches
# ---------------------------------------------------------------------------

_patch_wan_resample_rep_sentinel()
_patch_wan_resample_avoid_4d_fold()
_disable_tt_torch_function_override()

N_RUNS = 3

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_vae_decoder_480p():
    _run(resolution="480p", sharded=False)


def test_vae_decoder_720p():
    _run(resolution="720p", sharded=False)


def test_vae_decoder_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_vae_decoder_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(
        optimization_level=1,
        experimental_enable_dram_space_saving_optimization=True,
        export_path="model",
        export_model_name="vae_decoder",
        enable_trace=True,
    )
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]

    wrapper = VAEDecoderWrapper(load_vae()).eval().bfloat16()

    z = torch.randn(
        1,
        LATENT_CHANNELS,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    mesh = wan22_mesh() if sharded else None
    use_sharding = sharded and len(mesh.device_ids) > 1
    if use_sharding:
        enable_spmd()

    device = xm.xla_device()
    torch_xla.set_custom_compile_options(compiler_config.to_torch_compile_options())

    wrapper_on_device = wrapper.to(device)
    inputs_on_device = [z.to(device)]

    if use_sharding:
        for tensor, spec in shard_vae_decoder_specs(wrapper_on_device.vae).items():
            xs.mark_sharding(tensor, mesh, spec)

    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        warmup_start = time.perf_counter_ns()
        with safe_xla_slicing():
            out = compiled(*inputs_on_device)
            out_cpu = out.to("cpu")
        warmup_end = time.perf_counter_ns()

        warm_times = []
        for _ in range(N_RUNS):
            warm_start = time.perf_counter_ns()
            with safe_xla_slicing():
                tt_out = compiled(*inputs_on_device)
                tt_out = tt_out.to("cpu")
            warm_end = time.perf_counter_ns()
            warm_times.append(warm_end - warm_start)

    wrapper_cpu = wrapper_on_device.to("cpu")
    with torch.no_grad():
        cpu_out = wrapper_cpu(z)

    pcc = compute_pcc(tt_out, cpu_out)

    warmup_ms = (warmup_end - warmup_start) / 1e6
    warm_times_ms = [t / 1e6 for t in warm_times]

    print("====================================================================")
    print(f"| PERF: vae_decoder {resolution} {'sharded' if sharded else 'single'}")
    print("--------------------------------------------------------------------")
    print(f"| cold (compile + run) e2e: {warmup_ms:.4f} ms")
    print(f"| warm times: {', '.join(f'{t:.4f} ms' for t in warm_times_ms)}")
    print(f"| PCC: {pcc}")
    print("====================================================================")
