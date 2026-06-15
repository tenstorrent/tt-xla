# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Performance benchmarks for the Wan 2.2 TI2V-5B model components:
  - UMT5-XXL text encoder
  - 3D Causal VAE encoder / decoder (TI2V VAE, z_dim=48, spatial scale 16)
  - WanDiT (5B transformer, 30 blocks, in/out 48 channels, per-patch timestep)

All loaders, wrappers, sharding and monkey patches come from the 5B correctness
suite in tests/torch/models/wan5b/. The patches/contexts mirror that suite's
component tests exactly. Run logic lives in ``wan_common``; this file only
declares the family manifest and the (discoverable) test functions.
"""

import pytest
import wan_common
from wan_common import SHARDING_VARIANTS, VARIANTS, WanFamily, per_patch_timestep

from tests.torch.models.wan5b import shared
from tests.torch.models.wan5b.monkey_patch import (
    _patch_wan_resample_rep_sentinel,
    _patch_wan_time_embedder_dtype_probe,
    safe_xla_slicing,
    torch_function_override_disabled,
)

FAMILY = WanFamily(
    name_prefix="Wan2.2-TI2V-5B",
    shared=shared,
    dit_patches=(_patch_wan_time_embedder_dtype_probe,),
    vae_decoder_patches=(_patch_wan_resample_rep_sentinel,),
    dit_timestep=per_patch_timestep,
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
