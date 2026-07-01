# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Infinity 2B — transformer component test."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.utilities.torch_multichip_utils import get_mesh

from third_party.tt_forge_models.infinity.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="NameError: cannot access free variable 'named_children' where it is not associated with a value in enclosing scope - https://github.com/tenstorrent/tt-xla/issues/5019"
)
@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.model_test
def test_transformer():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.INFINITY_2B)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
