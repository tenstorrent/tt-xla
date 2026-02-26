# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

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


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <system_desc_path>")
        sys.exit(1)

    main(sys.argv[1])
