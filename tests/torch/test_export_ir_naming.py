# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""E2E test for IR export functionality."""

import re
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch_xla
import torch_xla.runtime as xr


@pytest.mark.push
@pytest.mark.single_device
def test_export_ir_naming():
    """Test that IR export generates MLIR files with custom model name in format <stage>_<model>_g<N>_<timestamp>.mlir."""
    xr.set_device_type("TT")
    device = torch_xla.device()

    with tempfile.TemporaryDirectory() as export_dir:
        model_name = "test_mlp"
        torch_xla.set_custom_compile_options({
            "export_path": export_dir,
            "export_model_name": model_name,
        })

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        ).to(torch.bfloat16).to(device)

        x = torch.randn(4, 64, dtype=torch.bfloat16, device=device)

        # Forward (g0) + Backward (g1)
        out = model(x)
        torch_xla.sync()
        out.sum().backward()
        torch_xla.sync()

        # Verify IR directory exists
        irs_dir = Path(export_dir) / "irs"
        assert irs_dir.exists(), f"IR directory not created at {irs_dir}"

        mlir_files = list(irs_dir.glob("*.mlir"))

        # Verify each stage has files for g0 (forward) and g1 (backward)
        # Pattern: <stage>_<model_name>_g<N>_<timestamp>.mlir
        expected_stages = ["vhlo", "shlo", "ttir", "ttnn"]
        for stage in expected_stages:
            for graph_num in [0, 1]:
                pattern = re.compile(rf"^{stage}_{model_name}_g{graph_num}_\d+\.mlir$")
                matching = [f for f in mlir_files if pattern.match(f.name)]
                assert len(matching) > 0, f"No file matching {stage}_{model_name}_g{graph_num}_<timestamp>.mlir"
