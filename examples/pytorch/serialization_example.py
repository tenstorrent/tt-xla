# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def run_serialization_example():
    """Run model and serialize compiled artifacts to disk."""
    cache_dir = f"{os.getcwd()}/cachedir"
    xr.initialize_cache(cache_dir)

    device = xm.xla_device()
    model = SimpleModel().to(device)
    x = torch.randn(3, 4).to(device)
    y = torch.randn(3, 4).to(device)
    output = model(x, y)
    output.to("cpu")

    parse_compiled_artifacts_from_cache_to_disk(cache_dir, "output/model")

    return cache_dir


def test_serialization_example():
    """Test that serialization creates expected output files."""
    xr.set_device_type("TT")

    output_dir = Path("output")
    cache_dir = f"{os.getcwd()}/cachedir"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    cache_dir = run_serialization_example()

    expected_files = [
        Path("output/model.ttnn"),
        Path("output/model_ttnn.mlir"),
        Path("output/model_ttir.mlir"),
    ]

    try:
        for filepath in expected_files:
            assert filepath.exists(), f"Expected file {filepath} was not created"
        print("All expected files were created successfully.")
    finally:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)


# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    xr.set_device_type("TT")

    cache_dir = run_serialization_example()
    print(f"Artifacts serialized to output/model.*")
