# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev transformer — bfp8 diagnostic #2: bfp_bf8 weights WITHOUT dram_space_saving.

Isolates whether the bfp8 segfault is caused by interaction with the
dram_space_saving pass (cause #2) vs. the weight-conversion pass alone.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
def test_transformer_fitB_bfp8_nodram():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.TRANSFORMER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
        compiler_config=CompilerConfig(
            experimental_weight_dtype="bfp_bf8",
        ),
    )
