#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for GPT-OSS 120B with sparse MLP - can be run standalone or as pytest.
"""

import os
import sys
from pathlib import Path

_repo_root = Path(__file__).parent
sys.path.insert(0, str(_repo_root / "tests"))

import numpy as np
import torch
import torch_xla.core.xla_model as xm
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.comparators.comparison_config import ComparisonConfig, PccConfig
from infra.comparators.torch_comparator import TorchComparator
from torch_xla.distributed.spmd import Mesh
from tt_torch.sparse_mlp import enable_sparse_mlp

from third_party.tt_forge_models.gpt_oss.pytorch.loader import ModelLoader, ModelVariant


def test_gpt_oss_120b_sparse():
    """Test GPT-OSS 120B with sparse MLP."""
    print("Loading model...")
    loader = ModelLoader(variant=ModelVariant.GPT_OSS_120B)
    model = loader.load_model()
    inputs = loader.load_inputs()

    print("Enabling sparse MLP...")
    model = enable_sparse_mlp(model, verbose=True)

    print("Running golden reference on CPU...")
    with torch.no_grad():
        golden = model(
            **{
                k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
        )

    print("Setting up XLA for TT device...")
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    xr.use_spmd()

    print("Moving model to XLA device...")
    model = model.to(xm.xla_device())

    num_devices = xr.global_runtime_device_count()
    mesh_shape, axis_names = loader.get_mesh_config(num_devices)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, axis_names)

    print(f"Applying shard specs for {num_devices} devices...")
    shard_specs = loader.load_shard_spec(model)
    for tensor, shard_spec in shard_specs.items():
        xs.mark_sharding(tensor, mesh, shard_spec)

    print("Compiling and running on TT device...")
    inputs_device = {
        k: v.to(xm.xla_device()) if isinstance(v, torch.Tensor) else v
        for k, v in inputs.items()
    }

    compiled = torch.compile(model, backend="tt")
    with torch.no_grad():
        output = compiled(**inputs_device)

    xm.mark_step()
    xm.wait_device_ops()

    print("Comparing outputs...")
    config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
    comparator = TorchComparator(config)
    result = comparator.compare(output.logits.cpu(), golden.logits.cpu())

    print(f"\nPCC: {result.pcc:.6f}")
    print(f"ATOL: {result.atol:.6e}")
    print(f"Passed: {result.passed}")

    assert result.passed, f"PCC comparison failed: {result.error_message}"
    print("✅ Test passed!")


if __name__ == "__main__":
    test_gpt_oss_120b_sparse()
