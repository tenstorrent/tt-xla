# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compile-only test: prove the CPU compile-only feature works without any TT
hardware.

The test compiles a simple model against a checked-in system descriptor
(`*_system_desc.ttsys`) instead of opening a real chip. Producing a descriptor
needs hardware (the `save` path reads a live chip); compiling against one does
not. So a descriptor is generated on each arch's hardware and checked into
`system_descs/`, and this test compiles against every one of them on CPU.

Regenerate a descriptor by running this file directly on the target hardware:
    python test_compile_only.py save system_descs/<arch>
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
from ttxla_tools import save_system_descriptor_to_disk


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def _save_system_desc(output_prefix: str):
    """Save the system descriptor from live hardware to disk."""
    xr.set_device_type("TT")
    xm.xla_device()
    save_system_descriptor_to_disk(output_prefix)


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


def _discover_fixtures():
    """Parametrize over every checked-in system descriptor.

    Discovered at collection time, so a new arch is picked up automatically
    once its fixture is checked into system_descs/. If none exist yet, emit a
    single skipped param so the suite stays green rather than erroring.
    """
    # Checked-in `.ttsys` fixtures, one per arch, named `<arch>_system_desc.ttsys`.
    FIXTURE_DIR = Path(__file__).resolve().parent / "system_descs"

    fixtures = sorted(FIXTURE_DIR.glob("*_system_desc.ttsys"))
    if not fixtures:
        return [
            pytest.param(
                None,
                id="no-fixtures",
                marks=pytest.mark.skip(
                    reason=f"no *_system_desc.ttsys fixtures in {FIXTURE_DIR}"
                ),
            )
        ]
    # id = arch name (strip the `_system_desc.ttsys` suffix) for readable test ids.
    return [
        pytest.param(f, id=f.name.removesuffix("_system_desc.ttsys")) for f in fixtures
    ]


@pytest.mark.nightly
@pytest.mark.parametrize("system_desc_path", _discover_fixtures())
def test_compile_only_per_arch(system_desc_path, tmp_path):
    """CPU-only: compile a simple model against a checked-in `.ttsys` fixture.

    No TT hardware required -- TT_COMPILE_ONLY_SYSTEM_DESC tells the plugin to
    read the chip spec from a file instead of opening a real chip. Runs the
    compile step in a subprocess because the PJRT client cannot be reconfigured
    once initialized (one client per process).

    Compiled artifacts land in tmp_path/output and are auto-cleaned by pytest;
    they're disposable, the assertions are what prove the compile succeeded.
    """
    script = str(Path(__file__).resolve())
    subprocess.run(
        [sys.executable, script, "compile", str(system_desc_path)],
        cwd=tmp_path,
        check=True,
    )

    _assert_compiled_artifacts(tmp_path / "output", label=system_desc_path.name)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "save":
        _save_system_desc(sys.argv[2])
    elif len(sys.argv) == 3 and sys.argv[1] == "compile":
        _compile_only(sys.argv[2])
    else:
        print(f"Usage: {sys.argv[0]} save <output_prefix>")
        print(f"       {sys.argv[0]} compile <system_desc_path>")
        sys.exit(1)
