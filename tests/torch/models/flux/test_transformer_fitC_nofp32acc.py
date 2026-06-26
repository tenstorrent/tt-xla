# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""FLUX.1-dev transformer — single-chip fit experiment C: fp32_dest_acc_en=False.

Precision-preserving DRAM lever. The baseline OOM trigger is a 132,120,576 B
(=33M x 4B = fp32) intermediate buffer; the model is bf16, so that tensor is
fp32 only because fp32 dest-accumulation is on by default. Disabling it should
store those intermediates as bf16 (halving them), targeting the 123 MiB gap.
"""

import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test

from tests.infra.testers.compiler_config import CompilerConfig
from third_party.tt_forge_models.flux.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
def test_transformer_fitC_nofp32acc():
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
            experimental_enable_dram_space_saving_optimization=True,
            fp32_dest_acc_en=False,
        ),
    )
