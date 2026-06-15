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
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

# One arch per checked-in descriptor. Each descriptor is generated on real
# hardware via save_system_desc.py and committed to system_descs/.
# Add a new arch here once its descriptor is committed.
ARCHS_WITH_SYSTEM_DESC = ["n150", "n300", "n300-llmbox", "p150"]


class SimpleModel(nn.Module):
    def forward(self, x, y):
        return x + y


def _cpu_compile_only(system_desc_path: str):
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
@pytest.mark.cpu
@pytest.mark.parametrize(
    "arch",
    ARCHS_WITH_SYSTEM_DESC,
)
def test_cpu_compile_only_per_arch(arch, tmp_path, monkeypatch):
    """CPU-only: compile a simple model against a checked-in `.ttsys` descriptor.

    No TT hardware required -- TT_COMPILE_ONLY_SYSTEM_DESC tells the plugin to
    read the chip spec from a file instead of opening a real chip.

    This test suite is run with --forked (one process per arch).
    The compile-only feature relies on torch_xla's singleton compilation cache,
    which does not permit output folder relocation or reuse in-process, so
    multiple archs are not able to be compiled in a single process.

    Compiled artifacts land in tmp_path/output and are auto-cleaned by pytest;
    they're disposable, the assertions are what prove the compile succeeded.
    """

    # Checked-in `<arch>_system_desc.ttsys` descriptors live in system_descs/.
    system_desc_dir = Path(__file__).resolve().parent / "system_descs"
    system_desc_path = system_desc_dir / f"{arch}_system_desc.ttsys"
    assert system_desc_path.exists(), f"missing system descriptor: {system_desc_path}"

    # Compile writes cache/ and output/ relative to cwd; keep them in tmp_path.
    monkeypatch.chdir(tmp_path)

    _cpu_compile_only(str(system_desc_path))

    _assert_compiled_artifacts(tmp_path / "output", label=arch)
