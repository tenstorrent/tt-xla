# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import jax
import numpy as np
from tt_jax import serialize_compiled_artifacts_to_disk
from ttxla_tools import save_system_descriptor_to_disk


def add(x, y):
    return x + y


def save_system_desc(output_prefix: str):
    """Save the system descriptor from live hardware to disk."""
    jax.devices("tt")
    save_system_descriptor_to_disk(output_prefix)


def compile_only(system_desc_path: str):
    """Compile for a target system using a saved system descriptor (no hardware needed)."""
    os.environ["TT_COMPILE_ONLY_SYSTEM_DESC"] = system_desc_path

    # Use numpy arrays, not jnp.array: jnp.array would eagerly dispatch to the
    # TT device and fail. serialize_compiled_artifacts only needs shape/dtype to
    # trace the function, so plain numpy arrays are sufficient.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    serialize_compiled_artifacts_to_disk(add, a, b, output_prefix="output/add")

    print("Artifacts written to output/add.*")
    print("To run on hardware: ttrt run output/add.ttnn")


def test_system_desc(tmp_path):
    """
    Save a system descriptor from hardware, then compile using it.

    Each step runs in its own subprocess because the PJRT client cannot be
    reconfigured once initialized.
    """
    script = str(Path(__file__).resolve())
    system_desc_path = str(tmp_path / "setup_system_desc.ttsys")

    subprocess.run(
        [sys.executable, script, "save", str(tmp_path / "setup")],
        check=True,
    )
    assert Path(system_desc_path).exists()

    subprocess.run(
        [sys.executable, script, "compile", system_desc_path],
        cwd=tmp_path,
    )

    for f in ["add.ttnn", "add_ttnn.mlir", "add_ttir.mlir"]:
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
