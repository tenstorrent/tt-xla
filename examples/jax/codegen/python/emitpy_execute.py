# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to execute generated Python code via EmitPy + PythonModelRunner.

Uses dry_run=False so the generated TTNN Python code is actually executed on
device and real output tensors are returned, which can be compared against a
CPU reference.
"""

import flax.nnx as nnx
import jax
import numpy as np


class Model(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs):
        self.A = nnx.Linear(32, 128, rngs=rngs)
        self.B = nnx.Linear(128, 64, rngs=rngs)

    def __call__(self, x):
        x = self.A(x)
        x = nnx.relu(x)
        x = self.B(x)
        return x


# Define forward pass.
def forward(graphdef, state, x):
    model = nnx.merge(graphdef, state)
    return model(x)


def main():
    # Initialize model on CPU.
    with jax.default_device(jax.devices("cpu")[0]):
        model = Model(rngs=nnx.Rngs(0))
        key = jax.random.key(1)
        x = jax.random.normal(key, (32, 32))
        graphdef, state = nnx.split(model)

        # Get CPU reference output.
        cpu_result = jax.jit(forward)(graphdef, state, x)

    # Run via EmitPy with actual execution (dry_run=False).
    # Inline the codegen_py logic to avoid importing tt_jax/tt_torch.
    compiler_options = {
        "backend": "codegen_py",
        "export_path": "model_execute",
        "export_tensors": True,
        "dry_run": False,
    }
    tt_result = jax.jit(forward, compiler_options=compiler_options)(graphdef, state, x)

    cpu_result_np = np.asarray(cpu_result)
    tt_result_np = np.asarray(tt_result)

    # Compare results.
    if tt_result is not None:
        if np.allclose(cpu_result_np, tt_result_np, atol=1e-2):
            print("SUCCESS: EmitPy result matches CPU reference.")
        else:
            max_diff = np.max(np.abs(cpu_result_np - tt_result_np))
            print(f"MISMATCH: max absolute difference = {max_diff}")
    else:
        print("ERROR: codegen_py returned None (expected tensor output).")


if __name__ == "__main__":
    main()
