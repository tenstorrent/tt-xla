# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to execute generated Python code via EmitPy + PythonModelRunner.

Uses dry_run=False so the generated TTNN Python code is actually executed on
device and real output tensors are returned.
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

        cpu_result = jax.jit(forward)(graphdef, state, x)

    compiler_options = {
        "backend": "codegen_py",
        "export_path": "model_execute",
        "export_tensors": True,
        "dry_run": False,
    }
    tt_result_emitpy = jax.jit(forward, compiler_options=compiler_options)(
        graphdef, state, x
    )
    tt_result_flatbuffer = jax.jit(forward)(graphdef, state, x)

    tt_result_np_emitpy = np.asarray(tt_result_emitpy)
    tt_result_np_flatbuffer = np.asarray(tt_result_flatbuffer)

    # Compare results.
    if np.allclose(tt_result_np_emitpy, tt_result_np_flatbuffer, atol=1e-2):
        print("SUCCESS: EmitPy result matches flatbuffer result.")
        return 0
    else:
        max_diff = np.max(np.abs(tt_result_np_emitpy - tt_result_np_flatbuffer))
        print(f"MISMATCH: max absolute difference = {max_diff}")
        return 1


def test_emitpy_execute():
    assert main() == 0


if __name__ == "__main__":
    main()
