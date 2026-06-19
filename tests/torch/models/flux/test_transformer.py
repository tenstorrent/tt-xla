# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — FluxTransformer2DModel component test (1024x1024 latent geometry)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


# Single-chip Blackhole DRAM OOM: the ~23.8 GB bf16 weights overflow DRAM during
# the weight load (LoadCachedOp->ToDeviceOp) - a 132 MB buffer can't allocate
# with <0.5 MB/bank free. Single-chip fit attempts that did NOT work:
#   - bf16 opt_level=1 and opt_level=2 (memory-layout / DRAM space-saving passes):
#     the optimizer compile is CPU-bound but does not converge in a practical
#     time (>2 h, no result) - impractical.
#   - bf16 experimental_enable_dram_space_saving_optimization / fp32_dest_acc_en
#     =False at opt_level=0: byte-identical OOM (these flags don't touch the
#     failing weight buffer).
#   - bfp8 weights (experimental_weight_dtype="bfp_bf8"): would halve weight
#     residency and fit, but segfaults in tt-metal pack_as_bfp_tiles during the
#     host float->bfp8 weight conversion (a prebuilt libtt_metal.so bug, reached
#     by both the compiler-option and apply_weight_dtype_overrides paths).
# Fix requires multichip tensor-parallel sharding (P300, 2x Blackhole) - deferred.
@pytest.mark.xfail(
    reason="Single-chip Blackhole DRAM OOM: ~23.8 GB bf16 weights overflow DRAM "
    "during weight load (132 MB buffer can't allocate, <0.5 MB/bank free). bf16 "
    "opt_level=1/2 compiles don't converge in a practical time; bfp8 weights "
    "segfault in tt-metal pack_as_bfp_tiles host packer. Needs multichip "
    "tensor-parallel sharding (deferred). Tracking issue #5251."
)
@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_transformer():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
