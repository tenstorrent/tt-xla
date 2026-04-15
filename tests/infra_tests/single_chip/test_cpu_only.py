# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk
from ttxla_tools import enable_compile_only, save_system_descriptor_to_disk


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def test_system_desc(tmp_path):
    """
    Save a system descriptor from hardware, then compile using it.

    Each step runs in its own subprocess because the PJRT client cannot be
    reconfigured once initialized.
    """
    script = str(Path(__file__).resolve())
    system_desc_path = str(tmp_path / "n150_system_desc.ttsys")

    subprocess.run(
        [sys.executable, script, "save", str(tmp_path / "setup")],
        check=True,
    )
    assert Path(system_desc_path).exists()

    subprocess.run(
        [sys.executable, script, "compile", system_desc_path],
        cwd=tmp_path,
    )

    for f in ["model.ttnn", "model_ttnn.mlir", "model_ttir.mlir"]:
        assert (
            tmp_path / "output" / f
        ).exists(), f"Expected output/{f} was not created"


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        save_system_desc(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "compile":
        compile_only(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} save <output_prefix>")
        print(f"       {sys.argv[0]} compile <system_desc_path>")
        sys.exit(1)
