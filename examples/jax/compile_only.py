# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import numpy as np
from tt_jax import save_system_descriptor_to_disk, serialize_compiled_artifacts_to_disk
from ttxla_tools import enable_compile_only


def add(x, y):
    return x + y


def main(system_desc_path: str):
    enable_compile_only(system_desc_path)

    # Use numpy arrays, not jnp.array: jnp.array would eagerly dispatch to the
    # TT device and fail. serialize_compiled_artifacts only needs shape/dtype to
    # trace the function, so plain numpy arrays are sufficient.
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])

    serialize_compiled_artifacts_to_disk(add, a, b, output_prefix="output/add")
    save_system_descriptor_to_disk("output/add")

    print("Artifacts written to output/add.*")
    print("To run on hardware: ttrt run output/add.ttnn")


def test_compile_only():
    """Save system descriptor from current hardware, then compile in compile-only mode."""
    import subprocess
    import tempfile

    import jax

    # Initialize PJRT with real hardware to obtain the system descriptor
    jax.devices("tt")

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
        assert os.path.exists(os.path.join(output_dir, "add_ttir.mlir"))
        assert os.path.exists(os.path.join(output_dir, "add_ttnn.mlir"))
        assert os.path.exists(os.path.join(output_dir, "add.ttnn"))
        assert os.path.exists(os.path.join(output_dir, "add_system_desc.ttsys"))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <system_desc_path>")
        sys.exit(1)

    main(sys.argv[1])
