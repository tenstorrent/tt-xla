# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev — T5 text encoder component test (sequence embedding)."""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.xfail(
    reason="PCC 0.933 < 0.99 on Blackhole (single-chip): compiles and runs e2e, "
    "numerical-accuracy gap only (close to historical flux ~0.935). On Wormhole "
    "it instead hits a tt-mlir compiler INTERNAL error (rank-0 scalar const-eval). "
    "Deferred per triage policy (PCC). Tracking issue TBD."
)
@pytest.mark.single_device
@pytest.mark.nightly
@pytest.mark.model_test
def test_t5_text_encoder():
    xr.set_device_type("TT")
    torch.manual_seed(42)

    loader = ModelLoader(ModelVariant.T5_TEXT_ENCODER)
    model = loader.load_model(dtype_override=torch.bfloat16)
    inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    run_graph_test(
        model,
        inputs,
        framework=Framework.TORCH,
    )
