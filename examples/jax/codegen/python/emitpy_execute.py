# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Demonstrates how to execute generated Python code via EmitPy + PythonModelRunner.
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

    # Run via EmitPy with actual execution (dry_run=False).
    # Inline the codegen_py logic to avoid importing tt_jax/tt_torch.
    compiler_options = {
        "backend": "codegen_py",
        "export_path": "model_execute",
        "export_tensors": True,
        "dry_run": False,
    }
    tt_result_codegen = jax.jit(forward, compiler_options=compiler_options)(
        graphdef, state, x
    )
    tt_result_fb = jax.jit(forward)(graphdef, state, x)

    tt_result_codegen_np = np.asarray(tt_result_codegen)
    tt_result_fb_np = np.asarray(tt_result_fb)

    # Compare results.
    if np.allclose(tt_result_codegen_np, tt_result_fb_np, atol=1e-2):
        print("SUCCESS: EmitPy result matches flatbuffer result.")
    else:
        max_diff = np.max(np.abs(tt_result_codegen_np - tt_result_fb_np))
        print(f"MISMATCH: max absolute difference = {max_diff}")


if __name__ == "__main__":
    main()
