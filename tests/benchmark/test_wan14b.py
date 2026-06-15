# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmarks for the Wan 2.2 A14B (I2V) model components:
  - UMT5-XXL text encoder (shared with the 5B model)
  - 3D Causal VAE encoder / decoder (Wan 2.1 VAE, z_dim=16, spatial scale 8)
  - WanDiT high-noise expert (40 blocks, in=36/out=16 channels, scalar timestep)

Covers the I2V-A14B high-noise expert (``transformer/``) only. All loaders,
wrappers, sharding and monkey patches come from the A14B correctness suite in
tests/torch/models/wan14b/. The patches/contexts mirror that suite's component
tests exactly. Run logic lives in ``wan_common``.
"""

import pytest
import wan_common
from wan_common import SHARDING_VARIANTS, VARIANTS, WanFamily, scalar_timestep

from tests.torch.models.wan14b import shared
from tests.torch.models.wan14b.monkey_patch import (
    _patch_wan_resample_rep_sentinel,
    _patch_wan_time_embedder_dtype_probe,
    safe_xla_slicing,
    torch_function_override_disabled,
)

FAMILY = WanFamily(
    name_prefix="Wan2.2-I2V-A14B",
    shared=shared,
    dit_patches=(_patch_wan_time_embedder_dtype_probe,),
    vae_decoder_patches=(_patch_wan_resample_rep_sentinel,),
    dit_timestep=scalar_timestep,
    safe_xla_slicing=safe_xla_slicing,
    dit_override_disabled=torch_function_override_disabled,
)


# UMT5 output is resolution-independent, so only the sharding axis is varied.
@pytest.mark.parametrize("sharded", SHARDING_VARIANTS)
def test_wan_umt5(sharded, output_file, request):
    wan_common.benchmark_umt5(FAMILY, sharded, output_file, request)


@pytest.mark.parametrize("resolution,sharded", VARIANTS)
def test_wan_vae_encoder(resolution, sharded, output_file, request):
    wan_common.benchmark_vae_encoder(FAMILY, resolution, sharded, output_file, request)


@pytest.mark.parametrize("resolution,sharded", VARIANTS)
def test_wan_vae_decoder(resolution, sharded, output_file, request):
    wan_common.benchmark_vae_decoder(FAMILY, resolution, sharded, output_file, request)


@pytest.mark.parametrize("resolution,sharded", VARIANTS)
def test_wan_dit(resolution, sharded, output_file, request):
    wan_common.benchmark_dit(FAMILY, resolution, sharded, output_file, request)
