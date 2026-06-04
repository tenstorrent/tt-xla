# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile-only test: prove the CPU compile-only feature works without any TT
hardware.

The test compiles a simple model against a checked-in system descriptor
(`*_system_desc.ttsys`) instead of opening a real chip. Producing a descriptor
needs hardware (a live chip read); compiling against one does not. So a
descriptor is generated on each arch's hardware and checked into
`system_descs/`, and this test compiles against every one of them on CPU.

Regenerate a descriptor by running save_system_desc.py on the target hardware.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def _compile_only(system_desc_path: str):
    """Compile for a target system using a saved system descriptor (no hardware needed)."""
    os.environ["TT_COMPILE_ONLY_SYSTEM_DESC"] = system_desc_path

    xr.set_device_type("TT")
    cache_dir = f"{os.getcwd()}/cachedir"
    xr.initialize_cache(cache_dir)

    device = xm.xla_device()
    model = SimpleModel().to(device)
    x = torch.randn(3, 4).to(device)
    y = torch.randn(3, 4).to(device)
    output = model(x, y)

    # torch_xla uses lazy execution: compilation and execution both happen here.
    # In compile-only mode, compilation succeeds and artifacts are cached.
    # Execution returns default-initialized output buffers (with uninitialized
    # data) instead of running on hardware.
    output.to("cpu")

    parse_compiled_artifacts_from_cache_to_disk(cache_dir, "output/model")


def _assert_compiled_artifacts(output_dir: Path, label: str = ""):
    """Assert compilation produced the expected artifacts."""
    # Artifacts the compile step is expected to emit.
    EXPECTED_ARTIFACTS = ["model.ttnn", "model_ttnn.mlir", "model_ttir.mlir"]
    suffix = f" for {label}" if label else ""
    for f in EXPECTED_ARTIFACTS:
        assert (output_dir / f).exists(), f"Expected {f} was not created{suffix}"


@pytest.mark.nightly
@pytest.mark.parametrize(
    # One arch per checked-in descriptor. Each descriptor is generated on real
    # hardware via save_system_desc.py and committed to system_descs/. Add a new
    # arch here once its descriptor is committed.
    "arch",
    [
        "n150",
        "n300",
        "n300-llmbox",
    ],
)
def test_compile_only_per_arch(arch, tmp_path):
    """CPU-only: compile a simple model against a checked-in `.ttsys` descriptor.

    No TT hardware required -- TT_COMPILE_ONLY_SYSTEM_DESC tells the plugin to
    read the chip spec from a file instead of opening a real chip. Runs the
    compile step in a subprocess because the PJRT client cannot be reconfigured
    once initialized (one client per process).

    Compiled artifacts land in tmp_path/output and are auto-cleaned by pytest;
    they're disposable, the assertions are what prove the compile succeeded.
    """
    # Checked-in `<arch>_system_desc.ttsys` descriptors live in system_descs/.
    system_desc_dir = Path(__file__).resolve().parent / "system_descs"
    system_desc_path = system_desc_dir / f"{arch}_system_desc.ttsys"
    assert system_desc_path.exists(), f"missing system descriptor: {system_desc_path}"

    script = str(Path(__file__).resolve())
    subprocess.run(
        [sys.executable, script, "compile", str(system_desc_path)],
        cwd=tmp_path,
        check=True,
    )

    _assert_compiled_artifacts(tmp_path / "output", label=arch)


if __name__ == "__main__":
    # Entry point for the compile subprocess spawned by the test itself. The
    # PJRT client can't be reconfigured once initialized, so each compile runs
    # in a fresh process. To save a descriptor instead, use save_system_desc.py.
    if len(sys.argv) == 3 and sys.argv[1] == "compile":
        _compile_only(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} compile <system_desc_path>")
        sys.exit(1)
