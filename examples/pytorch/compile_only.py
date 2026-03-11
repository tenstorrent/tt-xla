# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk
from ttxla_tools import enable_compile_only


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def main(system_desc_path: str):
    enable_compile_only(system_desc_path)

    xr.set_device_type("TT")
    cache_dir = f"{os.getcwd()}/cachedir"
    xr.initialize_cache(cache_dir)

    device = xm.xla_device()
    model = SimpleModel().to(device)
    x = torch.randn(3, 4).to(device)
    y = torch.randn(3, 4).to(device)
    output = model(x, y)

    # torch_xla uses lazy execution: compilation and execution both happen here.
    # In compile-only mode, compilation succeeds and artifacts are cached, but
    # execution raises an error since no hardware is available.
    try:
        output.to("cpu")
    except RuntimeError:
        pass

    parse_compiled_artifacts_from_cache_to_disk(cache_dir, "output/model")

    print("Artifacts written to output/model.*")
    print("To run on hardware: ttrt run output/model.ttnn")


def test_compile_only():
    """Save system descriptor from current hardware, then compile in compile-only mode."""
    import subprocess
    import tempfile

    from ttxla_tools import save_system_descriptor_to_disk

    # Initialize PJRT with real hardware to obtain the system descriptor
    xr.set_device_type("TT")
    xm.xla_device()

    with tempfile.TemporaryDirectory() as tmpdir:
        desc_prefix = os.path.join(tmpdir, "sys_desc")
        save_system_descriptor_to_disk(desc_prefix)
        desc_path = f"{desc_prefix}_system_desc.ttsys"
        assert os.path.exists(desc_path)

        # Run main() in a fresh subprocess so enable_compile_only() takes effect
        # before the PJRT client initializes.
        proc = subprocess.run(
            [sys.executable, __file__, desc_path],
            capture_output=True,
            text=True,
            cwd=tmpdir,
        )

        assert proc.returncode == 0, (
            f"compile_only failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
        )

        # Verify compilation artifacts were created
        output_dir = os.path.join(tmpdir, "output")
        assert os.path.exists(os.path.join(output_dir, "model_ttir.mlir"))
        assert os.path.exists(os.path.join(output_dir, "model_ttnn.mlir"))
        assert os.path.exists(os.path.join(output_dir, "model.ttnn"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <system_desc_path>")
        sys.exit(1)

    main(sys.argv[1])
