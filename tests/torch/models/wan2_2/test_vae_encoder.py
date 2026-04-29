# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B — 3D Causal VAE Encoder component test.

Encodes a single-frame image into the 48-channel latent used for I2V
conditioning.

IN:  (1, 3, 1, video_h, video_w) float
OUT: latent_dist.mean (1, 48, 1, latent_h, latent_w)
"""

import time

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd

from tests.infra.testers.compiler_config import CompilerConfig

from .shared import (
    RESOLUTIONS,
    VAEEncoderWrapper,
    compute_pcc,
    load_vae,
    shard_vae_encoder_specs,
    wan22_mesh,
)


def test_vae_encoder_480p():
    _run(resolution="480p", sharded=False)


def test_vae_encoder_720p():
    _run(resolution="720p", sharded=False)


def test_vae_encoder_480p_sharded():
    _run(resolution="480p", sharded=True)


def test_vae_encoder_720p_sharded():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(optimization_level=1)
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]

    wrapper = VAEEncoderWrapper(load_vae()).eval().bfloat16()

    x = torch.randn(1, 3, 1, shapes["video_h"], shapes["video_w"], dtype=torch.bfloat16)

    mesh = wan22_mesh() if sharded else None
    use_sharding = sharded and len(mesh.device_ids) > 1
    if use_sharding:
        enable_spmd()

    device = xm.xla_device()
    torch_xla.set_custom_compile_options(compiler_config.to_torch_compile_options())

    wrapper_on_device = wrapper.to(device)
    inputs_on_device = [x.to(device)]

    if use_sharding:
        for tensor, spec in shard_vae_encoder_specs(wrapper_on_device.vae).items():
            xs.mark_sharding(tensor, mesh, spec)

    compiled = torch.compile(wrapper_on_device, backend="tt")

    with torch.no_grad():
        warmup_start = time.perf_counter_ns()
        _ = compiled(*inputs_on_device)
        torch_xla.sync(wait=True)
        warmup_end = time.perf_counter_ns()

        warm_start = time.perf_counter_ns()
        tt_out = compiled(*inputs_on_device)
        torch_xla.sync(wait=True)
        warm_end = time.perf_counter_ns()

    tt_out_cpu = tt_out.to("cpu")

    wrapper_cpu = wrapper_on_device.to("cpu")
    with torch.no_grad():
        cpu_out = wrapper_cpu(x)

    pcc = compute_pcc(tt_out_cpu, cpu_out)

    warmup_ms = (warmup_end - warmup_start) / 1e6
    warm_ms = (warm_end - warm_start) / 1e6

    print("====================================================================")
    print(f"| PERF: vae_encoder {resolution} {'sharded' if sharded else 'single'}")
    print("--------------------------------------------------------------------")
    print(f"| warmup (compile + run) e2e: {warmup_ms:.4f} ms")
    print(f"| warm                   e2e: {warm_ms:.4f} ms")
    print(f"| PCC: {pcc}")
    print("====================================================================")
